import math

import pytest
import torch

from llsa.kernel.triton.mean_pool import mean_pool1d
from llsa.kernel.torch_op.flash_sparse_attention_res_2 import llsa_l2
from llsa.kernel.torch_op.flash_sparse_attention_res_2_varlen import llsa_l2_varlen
from llsa.kernel.triton.topk import compute_topk_indices_sparse, gen_sparse_indices


CUDA_AVAILABLE = torch.cuda.is_available()


def _make_level_weights(length: int, block_size: int, device: torch.device) -> torch.Tensor:
    p_ctx = math.ceil(length / block_size)
    weights = torch.full((p_ctx,), float(block_size), device=device, dtype=torch.float32)
    last_weight = float(length % block_size)
    if last_weight > 0:
        weights[-1] = last_weight
    return weights


def _torch_ref_llsa_l2(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk1: int, topk2: int, block_size: int,
) -> torch.Tensor:
    pq1 = mean_pool1d(q, block_size)
    pk1 = mean_pool1d(k, block_size)
    pv1 = mean_pool1d(v, block_size)

    pq2 = mean_pool1d(pq1, block_size)
    pk2 = mean_pool1d(pk1, block_size)
    pv2 = mean_pool1d(pv1, block_size)

    p_indices = gen_sparse_indices(pq2, pk2, topk2).to(torch.long)
    indices = compute_topk_indices_sparse(
        pq1, pk1, p_indices, topk1, p_indices.shape[-1], block_size,
    ).to(torch.long)

    bsz, n_head, m_ctx, d_head = q.shape
    n_ctx = k.shape[2]
    p_ctx1 = pk1.shape[2]
    p_ctx2 = pk2.shape[2]
    scale = 1.0 / math.sqrt(d_head)

    level1_weights = _make_level_weights(m_ctx, block_size, q.device)
    level2_weights = _make_level_weights(m_ctx, block_size * block_size, q.device)

    out = torch.empty_like(q)
    base = torch.arange(block_size, device=q.device, dtype=torch.long)

    for b in range(bsz):
        for h in range(n_head):
            for q_block_id in range(indices.shape[2]):
                q_start = q_block_id * block_size
                q_end = min(q_start + block_size, m_ctx)
                if q_start >= m_ctx:
                    continue

                q_block = q[b, h, q_start:q_end].to(torch.float32) * scale

                sparse_block_indices = indices[b, h, q_block_id]
                token_indices = (
                    sparse_block_indices[:, None] * block_size + base[None, :]
                ).reshape(-1)
                token_indices = token_indices[token_indices < n_ctx]

                prev_block_id = q_block_id // block_size
                level1_block_indices = p_indices[b, h, prev_block_id]
                level1_indices = (
                    level1_block_indices[:, None] * block_size + base[None, :]
                ).reshape(-1)
                level1_indices = level1_indices[level1_indices < p_ctx1]

                sparse_scores = None
                level1_scores = None
                if token_indices.numel() > 0:
                    sparse_k = k[b, h, token_indices].to(torch.float32)
                    sparse_v = v[b, h, token_indices].to(torch.float32)
                    sparse_scores = torch.matmul(q_block, sparse_k.transpose(-2, -1))
                else:
                    sparse_v = None

                if level1_indices.numel() > 0:
                    level1_k = pk1[b, h, level1_indices].to(torch.float32)
                    level1_v = pv1[b, h, level1_indices].to(torch.float32)
                    level1_scores = torch.matmul(q_block, level1_k.transpose(-2, -1))
                else:
                    level1_v = None

                level2_scores = torch.matmul(q_block, pk2[b, h].to(torch.float32).transpose(-2, -1))

                row_max = level2_scores.max(dim=-1).values
                if sparse_scores is not None:
                    row_max = torch.maximum(row_max, sparse_scores.max(dim=-1).values)
                if level1_scores is not None:
                    row_max = torch.maximum(row_max, level1_scores.max(dim=-1).values)

                denom = torch.zeros((q_end - q_start,), device=q.device, dtype=torch.float32)
                acc = torch.zeros((q_end - q_start, d_head), device=q.device, dtype=torch.float32)

                if sparse_scores is not None:
                    sparse_prob = torch.exp(sparse_scores - row_max[:, None])
                    denom = denom + sparse_prob.sum(dim=-1)
                    acc = acc + torch.matmul(sparse_prob, sparse_v)

                if level1_scores is not None:
                    level1_prob = torch.exp(level1_scores - row_max[:, None])
                    level1_prob = level1_prob * level1_weights[level1_indices][None, :]
                    denom = denom + level1_prob.sum(dim=-1)
                    acc = acc + torch.matmul(level1_prob, level1_v)

                level2_prob = torch.exp(level2_scores - row_max[:, None])
                level2_prob = level2_prob * level2_weights[None, :]
                denom = denom + level2_prob.sum(dim=-1)
                acc = acc + torch.matmul(level2_prob, pv2[b, h].to(torch.float32))

                out[b, h, q_start:q_end] = (acc / denom[:, None]).to(q.dtype)

    return out


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for Triton kernels")
@pytest.mark.parametrize(
    ("block_size", "seq_len", "topk1", "topk2"),
    [(16, 16 * 16 * 16 + 17, 8, 4), (64, 64 * 64 + 17, 8, 2)],
)
def test_llsa_l2_varlen_matches_torch_reference(block_size, seq_len, topk1, topk2):
    torch.manual_seed(0)
    dtype = torch.float32
    bsz, n_head, d_head = 1, 1, 32

    q = torch.randn(bsz, n_head, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, n_head, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(bsz, n_head, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)

    out = llsa_l2_varlen(q, k, v, topk1=topk1, topk2=topk2, block_size=block_size)
    grad = torch.randn_like(out)
    (out * grad).sum().backward()
    dq, dk, dv = q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref = _torch_ref_llsa_l2(q_ref, k_ref, v_ref, topk1=topk1, topk2=topk2, block_size=block_size)
    (out_ref * grad).sum().backward()

    torch.testing.assert_close(out, out_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dq, q_ref.grad, atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(dk, k_ref.grad, atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(dv, v_ref.grad, atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for Triton kernels")
def test_llsa_l2_varlen_matches_original_when_divisible():
    torch.manual_seed(1)
    dtype = torch.float32
    block_size = 16
    # The original l2 kernel only activates the level-2 residual path once
    # the second pooled level reaches a full 64-token block.
    seq_len = block_size * block_size * 64
    topk1, topk2 = 8, 4
    bsz, n_head, d_head = 1, 1, 32

    q = torch.randn(bsz, n_head, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, n_head, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(bsz, n_head, seq_len, d_head, device="cuda", dtype=dtype, requires_grad=True)

    out_new = llsa_l2_varlen(q, k, v, topk1=topk1, topk2=topk2, block_size=block_size)
    grad = torch.randn_like(out_new)
    (out_new * grad).sum().backward()
    dq_new = q.grad.detach().clone()
    dk_new = k.grad.detach().clone()
    dv_new = v.grad.detach().clone()

    q_old = q.detach().clone().requires_grad_(True)
    k_old = k.detach().clone().requires_grad_(True)
    v_old = v.detach().clone().requires_grad_(True)
    out_old = llsa_l2(q_old, k_old, v_old, topk1=topk1, topk2=topk2, block_size=block_size)
    (out_old * grad).sum().backward()

    torch.testing.assert_close(out_new, out_old, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq_new, q_old.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(dk_new, k_old.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(dv_new, v_old.grad, atol=4e-2, rtol=4e-2)
