from torch.autograd import Function

from ..triton.sparse_attention_res_2 import (sparse_indices_attn_res_2_2_fwd,
                                             sparse_indices_attn_res_scatter_2_2_bwd_v2,
                                             )
from ..triton.topk import gen_sparse_indices, compute_topk_indices_sparse
from ..triton.mean_pool import mean_pool1d


class FlashSparseResidualAttentionL2(Function):
    @staticmethod
    def forward(ctx, q, k, v, pq, pk, pv, pq2, pk2, pv2, topk, topk2, token_weight=1, token_weight_2=1, block_size=16):

        p_indices = gen_sparse_indices(pq2, pk2, topk2)
        indices = compute_topk_indices_sparse(
            pq, pk, p_indices, topk, topk2, block_size)

        output, m = sparse_indices_attn_res_2_2_fwd(q, k, v, pk, pv, pk2, pv2, indices,
                                                    p_indices, topk, topk2,
                                                    token_weight, token_weight_2,
                                                    int(block_size ** 0.5))

        ctx.save_for_backward(q, k, v, pk, pv, pk2, pv2, output, m,
                              indices, p_indices)
        ctx.topk = topk
        ctx.topk2 = topk2
        ctx.block_size = block_size
        ctx.token_weight = token_weight
        ctx.token_weight_2 = token_weight_2
        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, pk, pv, pk2, pv2, o, m, indices, p_indices = ctx.saved_tensors
        return *sparse_indices_attn_res_scatter_2_2_bwd_v2(do, q, k, v,
                                                           pk, pv, pk2, pv2, o, m,
                                                           indices, p_indices,
                                                           ctx.topk, ctx.topk2,
                                                           ctx.block_size,
                                                           ctx.token_weight,
                                                           ctx.token_weight_2), \
            None, None, None, None, None


flash_sparse_residual_attention_l2_op = FlashSparseResidualAttentionL2.apply


def llsa_l2(q, k, v, topk1=8, topk2=8, block_size=16):
    pq1 = mean_pool1d(q, block_size)
    pk1 = mean_pool1d(k, block_size)
    pv1 = mean_pool1d(v, block_size)

    pq2 = mean_pool1d(pq1, block_size)
    pk2 = mean_pool1d(pk1, block_size)
    pv2 = mean_pool1d(pv1, block_size)

    return flash_sparse_residual_attention_l2_op(q, k, v, pq1, pk1, pv1, pq2, pk2, pv2,
                                                 topk1,
                                                 topk2, block_size, block_size * block_size, block_size)
