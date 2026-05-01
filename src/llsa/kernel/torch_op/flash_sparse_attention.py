import torch
import torch.nn.functional as F
from torch.autograd import Function

from ..triton.sparse_attention import (sparse_indices_attn_fwd, sparse_indices_attn_scatter_bwd,
                                       sparse_indices_attn_scatter_bwd_v2,
                                       compute_indices_2)
from ..triton.topk import compute_topk_indices_sparse
from ..triton.indices_transpose import transpose_indices
from ..triton.mean_pool import mean_pool1d


def gen_sparse_indices(pq, pk, topk):
    score = torch.matmul(pq, pk.transpose(-2, -1))
    topk = min(topk, score.shape[-1])
    return torch.topk(score, topk,).indices


class FlashSparseAttention(Function):
    @staticmethod
    def forward(ctx, q, k, v, pq, pk, topk, block_size=16):
        indices = gen_sparse_indices(pq, pk, topk)
        output, m = sparse_indices_attn_fwd(
            q, k, v, indices, topk, int(block_size**0.5))
        ctx.save_for_backward(q, k, v, output, m, indices)
        ctx.topk = topk
        ctx.prev_size = pq.shape[2]
        ctx.block_size = block_size
        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, m, indices = ctx.saved_tensors

        # return *sparse_indices_attn_bwd(do, q, k, v, o, m, indices, ctx.topk, ctx.prev_size), None, None, None
        # return *sparse_indices_attn_scatter_bwd(do, q, k, v, o, m, indices, ctx.topk, ctx.block_size), None, None, None, None

        return *sparse_indices_attn_scatter_bwd_v2(do, q, k, v, o, m, indices, ctx.topk, ctx.block_size), None, None, None, None


flash_sparse_attention_op = FlashSparseAttention.apply


def sa_l1(q, k, v, topk=8, block_size=16):
    pq = mean_pool1d(q, block_size)
    pk = mean_pool1d(k, block_size)

    return flash_sparse_attention_op(q, k, v, pq, pk,
                                     topk, block_size)


class FlashSparseAttentionL2(Function):
    @staticmethod
    def forward(ctx, q, k, v, pq, pk, topk, pq2, pk2, topk2, block_size=16):
        p_indices = gen_sparse_indices(pq2, pk2, topk2)
        # p_indices = compute_topk_indices(pq2, pk2, topk2)

        # indices = compute_indices_2(
        #     pq, pk, topk, p_indices, 16)
        indices = compute_topk_indices_sparse(
            pq, pk, p_indices, topk, topk2, block_size)
        # indices = compute_topk_indices_sparse_varlen(
        #     pq, pk, p_indices, topk, topk2, block_size)

        # indices = gen_sparse_indices(pq, p2, topk)

        output, m = sparse_indices_attn_fwd(q, k, v, indices, topk)
        ctx.save_for_backward(q, k, v, output, m, indices)
        ctx.topk = topk
        ctx.prev_block_size = block_size
        return output

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, m, indices = ctx.saved_tensors
        # return *sparse_indices_attn_bwd(do, q, k, v, o, m, indices, ctx.topk, ctx.prev_size), None, None, None
        # return *sparse_indices_attn_scatter_bwd(do, q, k, v, o, m, indices, ctx.topk,
        #                                         ctx.prev_block_size), None, None, None, None, None, None
        # return *sparse_indices_attn_scatter_bwd(do, q, k, v, o, m, indices, ctx.topk,
        #                                         ctx.prev_block_size), None, None, None, None, None, None
        return *sparse_indices_attn_scatter_bwd_v2(do, q, k, v, o, m, indices, ctx.topk,
                                                   ctx.prev_block_size), None, None, None, None, None, None


flash_sparse_attention_l2_op = FlashSparseAttentionL2.apply
