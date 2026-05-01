import math

import pytest
import torch

from llsa.kernel.triton.mean_pool import mean_pool1d
from llsa.kernel.triton.topk import gen_sparse_indices
from llsa.kernel.torch_op.flash_sparse_attention_res_1 import llsa_l1
from llsa.kernel.torch_op.flash_sparse_attention_res_1_varlen import llsa_l1_varlen


CUDA_AVAILABLE = torch.cuda.is_available()


def _torch_ref_llsa_l1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, topk: int, block_size: int) -> torch.Tensor:
    pq = mean_pool1d(q, block_size)
    pk = mean_pool1d(k, block_size)
    pv = mean_pool1d(v, block_size)
    indices = gen_sparse_indices(pq, pk, topk).to(torch.long)

    bsz, n_head, m_ctx, d_head = q.shape
    n_ctx = k.shape[2]
    p_ctx = pk.shape[2]
    scale = 1.0 / math.sqrt(d_head)

    token_weight = float(block_size)
    block_last_weight = float(m_ctx % block_size)
    coarse_weights = torch.full(
        (p_ctx,), token_weight, device=q.device, dtype=torch.float32)
    if block_last_weight > 0:
        coarse_weights[-1] = block_last_weight

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

                block_indices = indices[b, h, q_block_id]
                token_indices = (
                    block_indices[:, None] * block_size + base[None, :]).reshape(-1)
                token_indices = token_indices[token_indices < n_ctx]

                coarse_scores = torch.matmul(
                    q_block, pk[b, h].to(torch.float32).transpose(-2, -1))
                if token_indices.numel() > 0:
                    sparse_k = k[b, h, token_indices].to(torch.float32)
                    sparse_v = v[b, h, token_indices].to(torch.float32)
                    sparse_scores = torch.matmul(
                        q_block, sparse_k.transpose(-2, -1))
                    row_max = torch.maximum(
                        sparse_scores.max(dim=-1).values,
                        coarse_scores.max(dim=-1).values,
                    )
                    sparse_prob = torch.exp(sparse_scores - row_max[:, None])
                else:
                    sparse_v = None
                    sparse_prob = None
                    row_max = coarse_scores.max(dim=-1).values

                coarse_prob = torch.exp(coarse_scores - row_max[:, None])
                coarse_prob = coarse_prob * coarse_weights[None, :]

                denom = coarse_prob.sum(dim=-1)
                acc = torch.matmul(coarse_prob, pv[b, h].to(torch.float32))

                if sparse_prob is not None:
                    denom = denom + sparse_prob.sum(dim=-1)
                    acc = acc + torch.matmul(sparse_prob, sparse_v)

                out[b, h, q_start:q_end] = (acc / denom[:, None]).to(q.dtype)

    return out


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
@pytest.mark.parametrize('block_size', [16])
def test_llsa_l1_varlen_matches_torch_reference(block_size):
    torch.manual_seed(0)
    dtype = torch.float32
    bsz, n_head, d_head = 1, 1, 32
    seq_len = block_size * block_size * block_size + 17
    topk = 4

    q = torch.randn(bsz, n_head, seq_len, d_head, device='cuda',
                    dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, n_head, seq_len, d_head, device='cuda',
                    dtype=dtype, requires_grad=True)
    v = torch.randn(bsz, n_head, seq_len, d_head, device='cuda',
                    dtype=dtype, requires_grad=True)

    out = llsa_l1_varlen(q, k, v, topk=topk, block_size=block_size)
    grad = torch.randn_like(out)
    (out * grad).sum().backward()
    dq, dk, dv = q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone()

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref = _torch_ref_llsa_l1(
        q_ref, k_ref, v_ref, topk=topk, block_size=block_size)
    (out_ref * grad).sum().backward()

    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dq, q_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(dk, k_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(dv, v_ref.grad, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
@pytest.mark.parametrize('block_size', [16])
def test_llsa_l1_varlen_matches_original_when_divisible(block_size):
    torch.manual_seed(1)
    dtype = torch.float32
    bsz, n_head, d_head = 1, 1, 32
    seq_len = block_size * block_size * block_size
    topk = 4

    q = torch.randn(bsz, n_head, seq_len, d_head, device='cuda',
                    dtype=dtype, requires_grad=True)
    k = torch.randn(bsz, n_head, seq_len, d_head, device='cuda',
                    dtype=dtype, requires_grad=True)
    v = torch.randn(bsz, n_head, seq_len, d_head, device='cuda',
                    dtype=dtype, requires_grad=True)

    out_new = llsa_l1_varlen(q, k, v, topk=topk, block_size=block_size)
    grad = torch.randn_like(out_new)
    (out_new * grad).sum().backward()
    dq_new, dk_new, dv_new = q.grad.detach().clone(
    ), k.grad.detach().clone(), v.grad.detach().clone()

    q_old = q.detach().clone().requires_grad_(True)
    k_old = k.detach().clone().requires_grad_(True)
    v_old = v.detach().clone().requires_grad_(True)
    out_old = llsa_l1(q_old, k_old, v_old, topk=topk, block_size=block_size)
    (out_old * grad).sum().backward()

    torch.testing.assert_close(out_new, out_old, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dq_new, q_old.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dk_new, k_old.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(dv_new, v_old.grad, atol=3e-2, rtol=3e-2)
