from torch.autograd import Function

from ..triton.mean_pool import mean_pool1d
from ..triton.topk import gen_sparse_indices
from ..triton.sparse_attention_res_1_varlen import (
    sparse_indices_attn_res_varlen_fwd,
    sparse_indices_attn_res_bwd_v2_varlen,
)


class FlashSparseResidualAttentionL1Varlen(Function):
    @staticmethod
    def forward(ctx, q, k, v, pq, pk, pv, topk, token_weight=1, block_size=16, block_last_weight=0):
        indices = gen_sparse_indices(pq, pk, topk)
        topk_eff = indices.shape[-1]
        output, m = sparse_indices_attn_res_varlen_fwd(
            q, k, v, pk, pv, indices,
            topk_eff, token_weight, block_last_weight,
            int(block_size ** 0.5),
        )

        ctx.save_for_backward(q, k, v, pk, pv, output, m, indices)
        ctx.topk = topk_eff
        ctx.block_size = block_size
        ctx.token_weight = token_weight
        ctx.block_last_weight = block_last_weight
        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, pk, pv, o, m, indices = ctx.saved_tensors
        return *sparse_indices_attn_res_bwd_v2_varlen(
            do, q, k, v, pk, pv, o, m, indices,
            ctx.topk,
            ctx.block_size,
            ctx.token_weight,
            ctx.block_last_weight,
        ), None, None, None, None, None


flash_sparse_residual_attention_l1_varlen_op = FlashSparseResidualAttentionL1Varlen.apply


def llsa_l1_varlen(q, k, v, topk=8, block_size=16):
    pq = mean_pool1d(q, block_size)
    pk = mean_pool1d(k, block_size)
    pv = mean_pool1d(v, block_size)

    block_last_weight = q.shape[2] % block_size
    return flash_sparse_residual_attention_l1_varlen_op(
        q, k, v, pq, pk, pv,
        topk,
        block_size,
        block_size,
        block_last_weight,
    )
