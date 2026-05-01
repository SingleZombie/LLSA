import pytest
import torch
import torch.nn.functional as F
import triton

from llsa.kernel.triton.mean_pool import mean_pool1d


CUDA_AVAILABLE = torch.cuda.is_available()


def _torch_ref_mean_pool_bhsc(x: torch.Tensor, k: int) -> torch.Tensor:
    b, h, s, c = x.shape
    y = x.permute(0, 1, 3, 2).reshape(b * h, c, s)
    y = F.avg_pool1d(y, kernel_size=k, stride=k,
                     ceil_mode=True, count_include_pad=False)
    y = y.reshape(b, h, c, y.shape[-1]).permute(0, 1, 3, 2).contiguous()
    return y


def _torch_ref_mean_pool_bsc(x: torch.Tensor, k: int) -> torch.Tensor:
    y = x.permute(0, 2, 1)
    y = F.avg_pool1d(y, kernel_size=k, stride=k,
                     ceil_mode=True, count_include_pad=False)
    y = y.permute(0, 2, 1).contiguous()
    return y


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('seq_len,k', [(1, 4), (7, 4), (16, 4), (17, 8), (33, 16), (65, 32)])
def test_mean_pool_bhsc_forward(dtype, seq_len, k):
    b, h, c = 2, 3, 32
    x = torch.randn(b, h, seq_len, c, device='cuda', dtype=dtype)

    y = mean_pool1d(x, k, format='bhsc')
    y_ref = _torch_ref_mean_pool_bhsc(x, k)

    atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('seq_len,k', [(1, 4), (7, 4), (16, 4), (17, 8), (33, 16), (65, 32)])
def test_mean_pool_bsc_forward(dtype, seq_len, k):
    b, c = 2, 130
    x = torch.randn(b, seq_len, c, device='cuda', dtype=dtype)

    y = mean_pool1d(x, k, format='bsc')
    y_ref = _torch_ref_mean_pool_bsc(x, k)

    atol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
@pytest.mark.parametrize('format_name', ['bhsc', 'bsc'])
@pytest.mark.parametrize('seq_len,k', [(7, 4), (17, 8), (33, 16)])
def test_mean_pool_backward(format_name, seq_len, k):
    dtype = torch.float32

    if format_name == 'bhsc':
        x = torch.randn(2, 3, seq_len, 16, device='cuda',
                        dtype=dtype, requires_grad=True)
        y = mean_pool1d(x, k, format='bhsc')
        grad = torch.randn_like(y)
        y.backward(grad)
        dx = x.grad.detach().clone()

        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = _torch_ref_mean_pool_bhsc(x_ref, k)
        y_ref.backward(grad)
        dx_ref = x_ref.grad
    else:
        x = torch.randn(2, seq_len, 67, device='cuda',
                        dtype=dtype, requires_grad=True)
        y = mean_pool1d(x, k, format='bsc')
        grad = torch.randn_like(y)
        y.backward(grad)
        dx = x.grad.detach().clone()

        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = _torch_ref_mean_pool_bsc(x_ref, k)
        y_ref.backward(grad)
        dx_ref = x_ref.grad

    torch.testing.assert_close(dx, dx_ref, atol=1e-5, rtol=1e-5)


def _bench_case(x: torch.Tensor, k: int, format_name: str):
    quantiles = [0.5, 0.2, 0.8]

    if format_name == 'bhsc':
        x_ref = x.permute(0, 1, 3, 2).reshape(
            x.shape[0] * x.shape[1], x.shape[3], x.shape[2])

        ms_triton, min_triton, max_triton = triton.testing.do_bench(
            lambda: mean_pool1d(x, k, format='bhsc'), quantiles=quantiles
        )
        ms_torch, min_torch, max_torch = triton.testing.do_bench(
            lambda: F.avg_pool1d(x_ref, kernel_size=k, stride=k,
                                 ceil_mode=True, count_include_pad=False),
            quantiles=quantiles,
        )
    else:
        x_ref = x.permute(0, 2, 1)

        ms_triton, min_triton, max_triton = triton.testing.do_bench(
            lambda: mean_pool1d(x, k, format='bsc'), quantiles=quantiles
        )
        ms_torch, min_torch, max_torch = triton.testing.do_bench(
            lambda: F.avg_pool1d(x_ref, kernel_size=k, stride=k,
                                 ceil_mode=True, count_include_pad=False),
            quantiles=quantiles,
        )

    return {
        'triton': (ms_triton, min_triton, max_triton),
        'torch': (ms_torch, min_torch, max_torch),
    }


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='CUDA is required for Triton kernels')
def test_benchmark_smoke():
    x = torch.randn(2, 4, 257, 64, device='cuda', dtype=torch.float16)
    result = _bench_case(x, k=16, format_name='bhsc')
    assert 'triton' in result and 'torch' in result


def run_benchmark():
    if not CUDA_AVAILABLE:
        print('CUDA not available, skip benchmark.')
        return

    print('format  dtype     S     k   triton_ms  torch_ms  speedup(torch/triton)')
    print('-' * 72)

    cases = [
        ('bhsc', torch.float16, 257, 16, (2, 4, 64)),
        ('bhsc', torch.bfloat16, 513, 32, (2, 4, 64)),
        ('bsc', torch.float16, 257, 130, (2, None, None)),
        ('bsc', torch.bfloat16, 513, 130, (2, None, None)),
    ]

    for format_name, dtype, seq_len, channels_or_k, shape_info in cases:
        if format_name == 'bhsc':
            b, h, c = shape_info
            k = channels_or_k
            x = torch.randn(b, h, seq_len, c, device='cuda', dtype=dtype)
        else:
            b = shape_info[0]
            c = channels_or_k
            k = 16 if seq_len < 400 else 32
            x = torch.randn(b, seq_len, c, device='cuda', dtype=dtype)

        result = _bench_case(x, k, format_name)
        triton_ms = result['triton'][0]
        torch_ms = result['torch'][0]
        speedup = torch_ms / triton_ms
        print(f"{format_name:5s}  {str(dtype).split('.')[-1]:8s} {seq_len:5d} {k:4d}"
              f" {triton_ms:10.4f} {torch_ms:9.4f} {speedup:20.3f}")


if __name__ == '__main__':
    # Select one mode:
    # 1) Unit tests: pytest -q tests/test_mean_pool.py
    # 2) Benchmark: uncomment the next line and run this file directly.
    # run_benchmark()

    # Default behavior when executed directly: run unit tests in this file.
    raise SystemExit(pytest.main([__file__, '-q']))
