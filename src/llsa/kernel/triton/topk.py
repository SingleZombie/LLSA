import torch
import triton
import triton.language as tl


def gen_sparse_indices(pq, pk, topk):
    score = torch.matmul(pq, pk.transpose(-2, -1))
    topk = min(topk, score.shape[-1])
    return torch.topk(score, topk,).indices


@triton.jit
def _topk_lastdim_k8_kernel(
    PQ, PK,
    OUT_I,
    n_ctx,
    stride_pq_b, stride_pq_h, stride_pq_m, stride_pq_d,
    stride_pk_b, stride_pk_h, stride_pk_n, stride_pk_d,
    stride_i_b, stride_i_h, stride_i_m, stride_i_k,
    NUM_HEAD: tl.constexpr,
    D_HEAD: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(1)
    pid_s = tl.program_id(0)

    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    offs_m = pid_s * BLOCK_M + tl.arange(0, BLOCK_M)
    pq_ptrs = PQ + cur_b * stride_pq_b + cur_h * stride_pq_h + offs_m * stride_pq_m
    pk_ptrs = PK + cur_b * stride_pk_b + cur_h * stride_pk_h

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    mask_pk = offs_n < n_ctx
    pq = tl.load(pq_ptrs[:, None] + offs_d[None, :] * stride_pq_d)
    pkT = tl.load(pk_ptrs + offs_n[None, :] *
                  stride_pk_n + offs_d[:, None] * stride_pk_d,
                  mask=mask_pk[None, :], other=float('-inf'))

    qk = tl.dot(pq, pkT)
    vals = qk.to(tl.float32)

    neg_inf = tl.full([1], float('-inf'), dtype=tl.float32)

    idxs = offs_n[None, :]

    k0 = tl.zeros([BLOCK_M, TOPK], dtype=tl.float32)
    i0 = tl.zeros([BLOCK_M, TOPK], dtype=tl.int32)
    indices = tl.arange(0, TOPK)[None, :]

    work = vals

    for t in tl.static_range(TOPK):
        maxv = tl.max(work, axis=1)[:, None]
        cand = tl.where(work == maxv, -idxs.to(tl.float32), neg_inf)
        arg_neg = tl.max(cand, axis=1)[:, None]
        arg = (-arg_neg).to(tl.int32)

        k0 = tl.where(indices == t, maxv, k0)
        i0 = tl.where(indices == t, arg, i0)

        work = tl.where(idxs == arg, neg_inf, work)

    base_i = OUT_I + cur_b * stride_i_b + \
        cur_h * stride_i_h + offs_m[:, None] * stride_i_m
    offs_k = tl.arange(0, TOPK)
    tl.store(base_i + offs_k[None, :] * stride_i_k, i0)


def compute_topk_indices(pq, pk, topk=8):
    B, H, M, D = pq.shape
    _, _, N, _ = pk.shape

    outi = torch.empty((B, H, M, topk), device=pq.device, dtype=torch.int32)

    BLOCK_N = N
    BLOCK_M = 4096 // BLOCK_N

    grid = (M // BLOCK_M, B*H)

    _topk_lastdim_k8_kernel[grid](
        pq, pk, outi,
        N,
        *pq.stride(),
        *pk.stride(),
        *outi.stride(),
        H, D, topk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N <= 128 else 8,
        num_stages=2,
    )

    return outi


@triton.jit
def _topk_indices_lastdim_k8_kernel(
    PQ, PK,
    P_INDICES,
    OUT_I,
    n_ctx,
    stride_pq_b, stride_pq_h, stride_pq_m, stride_pq_d,
    stride_pk_b, stride_pk_h, stride_pk_n, stride_pk_d,
    stride_i_b, stride_i_h, stride_i_m, stride_i_k,
    stride_pi_b, stride_pi_h, stride_pi_m, stride_pi_k,
    NUM_HEAD: tl.constexpr,
    D_HEAD: tl.constexpr,
    TOPK: tl.constexpr,
    P_TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(1)
    pid_s = tl.program_id(0)

    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    offs_m = BLOCK_SIZE * pid_s + tl.arange(0, BLOCK_SIZE)
    pq_ptrs = PQ + cur_b * stride_pq_b + cur_h * stride_pq_h + offs_m * stride_pq_m
    pk_ptrs = PK + cur_b * stride_pk_b + cur_h * stride_pk_h
    pi_ptrs = P_INDICES + cur_b * stride_pi_b + \
        cur_h * stride_pi_h + pid_s * stride_pi_m

    p_topk = tl.load(pi_ptrs + tl.arange(0, P_TOPK) * stride_pi_k)
    offs_n = p_topk[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    offs_n = offs_n.reshape([BLOCK_N,])

    offs_d = tl.arange(0, D_HEAD)

    pq = tl.load(pq_ptrs[:, None] + offs_d[None, :] * stride_pq_d)
    pkT = tl.load(pk_ptrs + offs_n[None, :] *
                  stride_pk_n + offs_d[:, None] * stride_pk_d)

    qk = tl.dot(pq, pkT)

    vals = qk.to(tl.float32)

    neg_inf = tl.full([1], float('-inf'), dtype=tl.float32)

    idxs = offs_n[None, :]

    k0 = tl.zeros([BLOCK_SIZE, TOPK], dtype=tl.float32)
    i0 = tl.zeros([BLOCK_SIZE, TOPK], dtype=tl.int32)
    indices = tl.arange(0, TOPK)[None, :]

    work = vals

    for t in tl.static_range(TOPK):
        maxv = tl.max(work, axis=1)[:, None]
        cand = tl.where(work == maxv, -idxs.to(tl.float32), neg_inf)
        arg_neg = tl.max(cand, axis=1)[:, None]
        arg = (-arg_neg).to(tl.int32)

        k0 = tl.where(indices == t, maxv, k0)
        i0 = tl.where(indices == t, arg, i0)

        work = tl.where(idxs == arg, neg_inf, work)

    base_i = OUT_I + cur_b * stride_i_b + \
        cur_h * stride_i_h + offs_m[:, None] * stride_i_m
    offs_k = tl.arange(0, TOPK)
    tl.store(base_i + offs_k[None, :] * stride_i_k, i0)


def compute_topk_indices_sparse(pq, pk, p_indices, topk=8, p_topk=8, block_size=16):
    B, H, M, D = pq.shape
    _, _, N, _ = pk.shape

    outi = torch.empty((B, H, M, topk), device=pq.device, dtype=torch.int32)

    BLOCK_SIZE = block_size
    BLOCK_N = p_topk * BLOCK_SIZE

    grid = (M // BLOCK_SIZE, B*H)

    _topk_indices_lastdim_k8_kernel[grid](
        pq, pk, p_indices, outi,
        N,
        *pq.stride(),
        *pk.stride(),
        *outi.stride(),
        *p_indices.stride(),
        H, D, topk, p_topk,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N == 128 else 8,
        num_stages=2,
    )

    return outi
