import torch
import triton

from llsa.kernel.triton.mean_pool import mean_pool1d
from llsa.kernel.torch_op.flash_sparse_attention_res_1 import flash_sparse_residual_attention_l1_op
from llsa.kernel.torch_op.flash_sparse_attention_res_2 import flash_sparse_residual_attention_l2_op


def llsa_l1(q, k, v, block_size=16, topk=8):
    pq = mean_pool1d(q, block_size)
    pk = mean_pool1d(k, block_size)
    pv = mean_pool1d(v, block_size)
    return flash_sparse_residual_attention_l1_op(q, k, v, pq, pk, pv, topk, block_size, block_size)


def llsa_l2(q, k, v, block_size=16, topk=8):
    pq = mean_pool1d(q, block_size)
    pk = mean_pool1d(k, block_size)
    pv = mean_pool1d(v, block_size)
    pq2 = mean_pool1d(pq, block_size)
    pk2 = mean_pool1d(pk, block_size)
    pv2 = mean_pool1d(pv, block_size)
    return flash_sparse_residual_attention_l2_op(q, k, v, pq, pk, pv, pq2, pk2, pv2, topk, topk,
                                                 block_size, block_size * block_size, block_size)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(7, 10)],
        line_arg='provider',
        line_vals=['l1', 'l2'],
        line_names=['LLSA_l1', 'LLSA_l2'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='attn-performance',
        args={},
    ))
def benchmark(size, provider):
    B, H, D = 2, 4, 64
    q = torch.randn(B, H, size*size, D, dtype=torch.bfloat16).cuda()
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'l1':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: llsa_l1(q, k, v), quantiles=quantiles)
    elif provider == 'l2':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: llsa_l2(q, k, v), quantiles=quantiles)

    return ms, max_ms, min_ms


if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True,
                  save_path='llsa_benchmark')
