from torch.autograd import Function

from ..triton.mean_pool import mean_pool1d
from ..triton.sparse_attention_res_2_varlen import (
    sparse_indices_attn_res_2_2_varlen_fwd,
    sparse_indices_attn_res_scatter_2_2_bwd_v2_varlen,
)
from ..triton.topk import compute_topk_indices_sparse, gen_sparse_indices


class FlashSparseResidualAttentionL2Varlen(Function):
    @staticmethod
    def forward(
        ctx, q, k, v, pq, pk, pv, pq2, pk2, pv2, topk, topk2,
        token_weight=1, token_weight_2=1, block_size=16,
        block_last_weight=0, block_last_weight_2=0,
    ):
        p_indices = gen_sparse_indices(pq2, pk2, topk2)
        topk2_eff = p_indices.shape[-1]
        indices = compute_topk_indices_sparse(
            pq, pk, p_indices, topk, topk2_eff, block_size,
        )
        topk_eff = indices.shape[-1]

        output, m = sparse_indices_attn_res_2_2_varlen_fwd(
            q, k, v, pk, pv, pk2, pv2, indices, p_indices,
            topk_eff, topk2_eff,
            token_weight, token_weight_2,
            block_last_weight, block_last_weight_2,
            int(block_size ** 0.5),
        )

        ctx.save_for_backward(q, k, v, pk, pv, pk2, pv2, output, m, indices, p_indices)
        ctx.topk = topk_eff
        ctx.topk2 = topk2_eff
        ctx.block_size = block_size
        ctx.token_weight = token_weight
        ctx.token_weight_2 = token_weight_2
        ctx.block_last_weight = block_last_weight
        ctx.block_last_weight_2 = block_last_weight_2
        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, pk, pv, pk2, pv2, o, m, indices, p_indices = ctx.saved_tensors
        return (
            *sparse_indices_attn_res_scatter_2_2_bwd_v2_varlen(
                do, q, k, v, pk, pv, pk2, pv2, o, m, indices, p_indices,
                ctx.topk, ctx.topk2, ctx.block_size,
                ctx.token_weight, ctx.token_weight_2,
                ctx.block_last_weight, ctx.block_last_weight_2,
            ),
            None, None, None, None, None, None, None,
        )


flash_sparse_residual_attention_l2_varlen_op = FlashSparseResidualAttentionL2Varlen.apply


def llsa_l2_varlen(q, k, v, topk1=8, topk2=8, block_size=16):
    pq1 = mean_pool1d(q, block_size)
    pk1 = mean_pool1d(k, block_size)
    pv1 = mean_pool1d(v, block_size)

    pq2 = mean_pool1d(pq1, block_size)
    pk2 = mean_pool1d(pk1, block_size)
    pv2 = mean_pool1d(pv1, block_size)

    block_last_weight = q.shape[2] % block_size
    block_last_weight_2 = q.shape[2] % (block_size * block_size)

    return flash_sparse_residual_attention_l2_varlen_op(
        q, k, v, pq1, pk1, pv1, pq2, pk2, pv2,
        topk1, topk2,
        block_size, block_size * block_size, block_size,
        block_last_weight, block_last_weight_2,
    )
