import pytest
import torch
import triton

from llsa.kernel.triton.topk import compute_topk_indices, compute_topk_indices_sparse


CUDA_AVAILABLE = torch.cuda.is_available()


def _torch_ref_topk_indices(pq: torch.Tensor, pk: torch.Tensor, topk: int) -> torch.Tensor:
    score = torch.matmul(pq, pk.transpose(-2, -1))
    k = min(topk, score.shape[-1])
    return torch.topk(score, k=k, dim=-1).indices.to(torch.int32)


def _torch_ref_topk_indices_sparse(
    pq: torch.Tensor,
    pk: torch.Tensor,
    p_indices: torch.Tensor,
    topk: int,
    block_size: int,
) -> torch.Tensor:
    bsz, n_head, m_ctx, _ = pq.shape
    _, _, n_ctx, _ = pk.shape

    m_blocks = m_ctx // block_size
    block_n = p_indices.shape[-1] * block_size
    k = min(topk, block_n)

    out = torch.empty((bsz, n_head, m_ctx, k),
                      device=pq.device, dtype=torch.int32)
    base = torch.arange(block_size, device=pq.device, dtype=torch.int64)

    for b in range(bsz):
        for h in range(n_head):
            for mb in range(m_blocks):
                cand_blocks = p_indices[b, h, mb].to(torch.int64)
                cand = (cand_blocks[:, None] *
                        block_size + base[None, :]).reshape(-1)
                cand = cand[cand < n_ctx]

                q_block = pq[b, h, mb * block_size:(mb + 1) * block_size]
                k_cand = pk[b, h, cand]
                score = torch.matmul(q_block, k_cand.transpose(-2, -1))

                top_local = torch.topk(score, k=k, dim=-1).indices
                top_global = cand[top_local].to(torch.int32)
                out[b, h, mb * block_size:(mb + 1) * block_size] = top_global

    return out


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
@pytest.mark.parametrize('topk', [1, 3, 6, 12])
def test_topk_dense_forward(topk):
    b, h, m, n, d = 2, 3, 32, 11, 64
    pq = torch.randn(b, h, m, d, device='cuda', dtype=torch.float32)
    pk = torch.randn(b, h, n, d, device='cuda', dtype=torch.float32)

    out = compute_topk_indices(pq, pk, topk=topk)
    score = torch.matmul(pq, pk.transpose(-2, -1))
    k = min(topk, n)
    ref_vals = torch.topk(score, k=k, dim=-1).values
    out_vals = torch.gather(score, dim=-1, index=out.to(torch.int64))

    assert out.dtype == torch.int32
    assert out.shape == (b, h, m, min(topk, n))
    torch.testing.assert_close(out_vals, ref_vals, atol=1e-2, rtol=2e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
@pytest.mark.parametrize('topk', [1, 5, 11, 13])
def test_topk_sparse_forward(topk):
    b, h, m, n, d = 2, 2, 64, 64, 48
    block_size = 16
    p_topk = 3

    assert m % block_size == 0
    assert n % block_size == 0

    m_blocks = m // block_size
    n_blocks = n // block_size

    pq = torch.randn(b, h, m, d, device='cuda', dtype=torch.float32)
    pk = torch.randn(b, h, n, d, device='cuda', dtype=torch.float32)
    p_indices = torch.argsort(
        torch.rand(b, h, m_blocks, n_blocks, device='cuda'), dim=-1
    )[..., :p_topk].to(torch.int32)

    out = compute_topk_indices_sparse(
        pq, pk, p_indices, topk=topk, p_topk=p_topk, block_size=block_size)
    ref = _torch_ref_topk_indices_sparse(
        pq, pk, p_indices, topk=topk, block_size=block_size)

    score = torch.matmul(pq, pk.transpose(-2, -1))
    out_vals = torch.gather(score, dim=-1, index=out.to(torch.int64))
    ref_vals = torch.gather(score, dim=-1, index=ref.to(torch.int64))

    assert out.dtype == torch.int32
    assert out.shape == (b, h, m, min(topk, p_topk * block_size))
    torch.testing.assert_close(out_vals, ref_vals, atol=5e-3, rtol=1e-3)


def _bench_dense(pq: torch.Tensor, pk: torch.Tensor, topk: int):
    quantiles = [0.5, 0.2, 0.8]
    ms_triton, min_triton, max_triton = triton.testing.do_bench(
        lambda: compute_topk_indices(pq, pk, topk=topk), quantiles=quantiles
    )
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: _torch_ref_topk_indices(pq, pk, topk=topk), quantiles=quantiles
    )
    return {
        'triton': (ms_triton, min_triton, max_triton),
        'torch': (ms_torch, min_torch, max_torch),
    }


def _bench_sparse(pq: torch.Tensor, pk: torch.Tensor, p_indices: torch.Tensor, topk: int, block_size: int):
    quantiles = [0.5, 0.2, 0.8]
    ms_triton, min_triton, max_triton = triton.testing.do_bench(
        lambda: compute_topk_indices_sparse(pq, pk, p_indices, topk=topk,
                                            p_topk=p_indices.shape[-1], block_size=block_size),
        quantiles=quantiles,
    )
    ms_torch, min_torch, max_torch = triton.testing.do_bench(
        lambda: _torch_ref_topk_indices_sparse(
            pq, pk, p_indices, topk=topk, block_size=block_size),
        quantiles=quantiles,
    )
    return {
        'triton': (ms_triton, min_triton, max_triton),
        'torch': (ms_torch, min_torch, max_torch),
    }


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
def test_topk_benchmark_smoke():
    pq = torch.randn(2, 2, 64, 64, device='cuda', dtype=torch.float16)
    pk = torch.randn(2, 2, 64, 64, device='cuda', dtype=torch.float16)

    dense = _bench_dense(pq, pk, topk=6)
    assert 'triton' in dense and 'torch' in dense

    block_size = 16
    p_topk = 4
    p_indices = torch.argsort(
        torch.rand(2, 2, 4, 4, device='cuda'), dim=-1
    )[..., :p_topk].to(torch.int32)
    sparse = _bench_sparse(pq, pk, p_indices, topk=7, block_size=block_size)
    assert 'triton' in sparse and 'torch' in sparse


def run_benchmark():
    if not CUDA_AVAILABLE:
        print('CUDA not available, skip benchmark.')
        return

    print('mode    dtype      topk   triton_ms  torch_ms  speedup(torch/triton)')
    print('-' * 72)

    pq = torch.randn(2, 2, 64, 64, device='cuda', dtype=torch.float16)
    pk = torch.randn(2, 2, 64, 64, device='cuda', dtype=torch.float16)

    for topk in [6, 11, 17]:
        dense = _bench_dense(pq, pk, topk=topk)
        triton_ms = dense['triton'][0]
        torch_ms = dense['torch'][0]
        speedup = torch_ms / triton_ms
        print(
            f"dense  {'float16':9s} {topk:5d} {triton_ms:10.4f} {torch_ms:9.4f} {speedup:20.3f}")

    seq_length = 512

    pq = torch.randn(2, 2, seq_length, 64, device='cuda', dtype=torch.float16)
    pk = torch.randn(2, 2, seq_length, 64, device='cuda', dtype=torch.float16)
    block_size = 16
    p_seq_length = seq_length // block_size
    p_topk = 4
    p_indices = torch.argsort(
        torch.rand(2, 2, p_seq_length, p_seq_length, device='cuda'), dim=-1
    )[..., :p_topk].to(torch.int32)

    for topk in [5, 9, 19]:
        sparse = _bench_sparse(
            pq, pk, p_indices, topk=topk, block_size=block_size)
        triton_ms = sparse['triton'][0]
        torch_ms = sparse['torch'][0]
        speedup = torch_ms / triton_ms
        print(
            f"sparse {'float16':9s} {topk:5d} {triton_ms:10.4f} {torch_ms:9.4f} {speedup:20.3f}")


if __name__ == '__main__':
    # Select one mode:
    # 1) Unit tests: pytest -q tests/test_topk.py
    # 2) Benchmark: uncomment the next line and run this file directly.
    # run_benchmark()

    # Default behavior when executed directly: run unit tests in this file.
    raise SystemExit(pytest.main([__file__, '-q']))
