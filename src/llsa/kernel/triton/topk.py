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
    m_ctx,
    n_ctx,
    d_head_out,
    topk_out,
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
    mask_m = offs_m < m_ctx
    pq_ptrs = PQ + cur_b * stride_pq_b + cur_h * stride_pq_h + offs_m * stride_pq_m
    pk_ptrs = PK + cur_b * stride_pk_b + cur_h * stride_pk_h

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    mask_d = offs_d < d_head_out

    mask_pk = offs_n < n_ctx
    pq = tl.load(pq_ptrs[:, None] + offs_d[None, :] * stride_pq_d,
                 mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    pkT = tl.load(pk_ptrs + offs_n[None, :] *
                  stride_pk_n + offs_d[:, None] * stride_pk_d,
                  mask=mask_pk[None, :] & mask_d[:, None], other=0.0)

    qk = tl.dot(pq, pkT)
    vals = qk.to(tl.float32)

    neg_inf = tl.full([1], float('-inf'), dtype=tl.float32)
    vals = tl.where(mask_pk[None, :], vals, neg_inf)

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
    store_mask = mask_m[:, None] & (offs_k[None, :] < topk_out)
    tl.store(base_i + offs_k[None, :] * stride_i_k, i0, mask=store_mask)


def compute_topk_indices(pq, pk, topk=8):
    B, H, M, D = pq.shape
    _, _, N, _ = pk.shape

    topk_out = min(topk, N)
    if topk_out <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    assert N <= 128, "This kernel is optimized for N <= 128. For larger N, consider using torch version."

    topk_kernel = triton.next_power_of_2(topk_out)
    d_head_kernel = triton.next_power_of_2(D)

    outi = torch.empty((B, H, M, topk_out),
                       device=pq.device, dtype=torch.int32)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_M = triton.next_power_of_2(max(1, 4096 // BLOCK_N))

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    _topk_lastdim_k8_kernel[grid](
        pq, pk, outi,
        M,
        N,
        D,
        topk_out,
        *pq.stride(),
        *pk.stride(),
        *outi.stride(),
        H, d_head_kernel, topk_kernel,
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
    m_ctx,
    n_ctx,
    d_head_out,
    topk_out,
    p_topk_out,
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
    mask_m = offs_m < m_ctx
    pq_ptrs = PQ + cur_b * stride_pq_b + cur_h * stride_pq_h + offs_m * stride_pq_m
    pk_ptrs = PK + cur_b * stride_pk_b + cur_h * stride_pk_h
    pi_ptrs = P_INDICES + cur_b * stride_pi_b + \
        cur_h * stride_pi_h + pid_s * stride_pi_m

    offs_pt = tl.arange(0, P_TOPK)
    mask_pt = offs_pt < p_topk_out
    p_topk = tl.load(pi_ptrs + offs_pt * stride_pi_k, mask=mask_pt, other=0)
    offs_n = p_topk[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_n = mask_pt[:, None] & (offs_n < n_ctx)
    offs_n = offs_n.reshape([BLOCK_N,])
    mask_n = mask_n.reshape([BLOCK_N,])

    offs_d = tl.arange(0, D_HEAD)
    mask_d = offs_d < d_head_out

    pq = tl.load(pq_ptrs[:, None] + offs_d[None, :] * stride_pq_d,
                 mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    pkT = tl.load(pk_ptrs + offs_n[None, :] *
                  stride_pk_n + offs_d[:, None] * stride_pk_d,
                  mask=mask_n[None, :] & mask_d[:, None], other=0.0)

    qk = tl.dot(pq, pkT)

    vals = qk.to(tl.float32)

    neg_inf = tl.full([1], float('-inf'), dtype=tl.float32)
    vals = tl.where(mask_n[None, :], vals, neg_inf)

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
    store_mask = mask_m[:, None] & (offs_k[None, :] < topk_out)
    tl.store(base_i + offs_k[None, :] * stride_i_k, i0, mask=store_mask)


def compute_topk_indices_sparse(pq, pk, p_indices, topk=8, p_topk=8, block_size=16):
    B, H, M, D = pq.shape
    _, _, N, _ = pk.shape

    BLOCK_SIZE = block_size
    assert BLOCK_SIZE >= 16, "BLOCK_SIZE must be at least 16"

    p_topk_out = min(p_topk, p_indices.shape[-1])
    if p_topk_out <= 0:
        raise ValueError(f"p_topk must be positive, got {p_topk}")

    p_topk_kernel = triton.next_power_of_2(p_topk_out)
    BLOCK_N = p_topk_kernel * BLOCK_SIZE

    topk_out = min(topk, p_topk_out * BLOCK_SIZE)
    if topk_out <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    topk_kernel = triton.next_power_of_2(topk_out)
    d_head_kernel = triton.next_power_of_2(D)

    outi = torch.empty((B, H, M, topk_out),
                       device=pq.device, dtype=torch.int32)

    grid = (triton.cdiv(M, BLOCK_SIZE), B * H)

    _topk_indices_lastdim_k8_kernel[grid](
        pq, pk, p_indices, outi,
        M,
        N,
        D,
        topk_out,
        p_topk_out,
        *pq.stride(),
        *pk.stride(),
        *outi.stride(),
        *p_indices.stride(),
        H, d_head_kernel, topk_kernel, p_topk_kernel,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_N=BLOCK_N,
        num_warps=4 if BLOCK_N == 128 else 8,
        num_stages=2,
    )

    return outi
