from torch.autograd import Function

from ..triton.sparse_attention_res_1 import (sparse_indices_attn_res_fwd,
                                             sparse_indices_attn_res_scatter_bwd_v2,
                                             )
from ..triton.topk import gen_sparse_indices


class FlashSparseResidualAttentionL1(Function):
    @staticmethod
    def forward(ctx, q, k, v, pq, pk, pv, topk, token_weight=1, block_size=16):

        indices = gen_sparse_indices(pq, pk, topk)

        output, m = sparse_indices_attn_res_fwd(q, k, v, pk, pv, indices,
                                                topk, token_weight,
                                                int(block_size ** 0.5))

        ctx.save_for_backward(q, k, v, pk, pv, output, m,
                              indices)
        ctx.topk = topk
        ctx.block_size = block_size
        ctx.token_weight = token_weight
        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, pk, pv, o, m, indices = ctx.saved_tensors
        return *sparse_indices_attn_res_scatter_bwd_v2(do, q, k, v,
                                                       pk, pv, o, m,
                                                       indices,
                                                       ctx.topk,
                                                       ctx.block_size, ctx.token_weight), \
            None, None, None, None, None, None


flash_sparse_residual_attention_l1_op = FlashSparseResidualAttentionL1.apply
