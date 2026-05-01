import math

import torch
import triton
import triton.language as tl

from .indices_transpose import transpose_indices


@triton.jit
def _sparse_indices_attn_res_2_2_varlen_fwd(
    Q, K, V, O, M, INDICES,
    PK, PV, PREV_INDICES,
    PK2, PV2,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_im, stride_ik,
    stride_mb, stride_mh, stride_mm,
    stride_pkb, stride_pkh, stride_pkn, stride_pkd,
    stride_pib, stride_pih, stride_pin, stride_pik,
    stride_pk2b, stride_pk2h, stride_pk2n, stride_pk2d,
    m_ctx, n_ctx, p_ctx, p_ctx2, NUM_HEAD: tl.constexpr,
    scale: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P: tl.constexpr,
    B_LENGTH: tl.constexpr,
    TOKEN_WEIGHT: tl.constexpr,
    TOKEN_WEIGHT_2: tl.constexpr,
    BLOCK_LAST_WEIGHT,
    BLOCK_LAST_WEIGHT_2,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    D_HEAD: tl.constexpr = 64,
):
    B_SIZE: tl.constexpr = B_LENGTH * B_LENGTH

    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    offsets_m = tl.arange(0, B_SIZE) + pid_m * B_SIZE
    offsets_d = tl.arange(0, D_HEAD)

    q_ptrs = (Q + cur_b * stride_qb + cur_h * stride_qh
              + offsets_m[:, None] * stride_qm
              + offsets_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offsets_m[:, None] < m_ctx, other=0.0)
    q *= scale

    m_i = tl.full([B_SIZE], -float("inf"), tl.float32)
    l_i = tl.zeros([B_SIZE], tl.float32)
    acc = tl.zeros([B_SIZE, D_HEAD], tl.float32)

    NUM_N_BLOCKS: tl.constexpr = BLOCK_N // B_SIZE
    for k_id in range(0, TOPK // NUM_N_BLOCKS):
        offsets_in = k_id * NUM_N_BLOCKS + tl.arange(0, NUM_N_BLOCKS)
        ind_ptrs = (INDICES + cur_b * stride_ib + cur_h * stride_ih
                    + pid_m * stride_im
                    + offsets_in * stride_ik)
        start_n = tl.load(ind_ptrs)
        offsets_n = start_n[:, None] * B_SIZE + tl.arange(0, B_SIZE)[None, :]
        offsets_n = offsets_n.reshape(BLOCK_N)

        mask_n = offsets_n < n_ctx
        k_ptrs = (K + cur_b * stride_kb + cur_h * stride_kh
                  + offsets_n[:, None] * stride_kn
                  + offsets_d[None, :] * stride_kd)
        v_ptrs = (V + cur_b * stride_vb + cur_h * stride_vh
                  + offsets_n[:, None] * stride_vn
                  + offsets_d[None, :] * stride_vd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32)
        scores = scores + tl.where(mask_n[None, :], 0.0, -1.0e6)
        m_ij = tl.max(scores, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        exp_scores = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(exp_scores, 1)
        l_i = l_i * tl.exp(m_i - m_i_new) + l_ij
        acc = acc * tl.exp(m_i - m_i_new)[:, None] + tl.dot(exp_scores.to(v.dtype), v)
        m_i = m_i_new

    prev_m_id = pid_m // B_SIZE
    for k_id in range(0, TOPK_P // NUM_N_BLOCKS):
        offsets_in = k_id * NUM_N_BLOCKS + tl.arange(0, NUM_N_BLOCKS)
        ind_ptrs = (PREV_INDICES + cur_b * stride_pib + cur_h * stride_pih
                    + prev_m_id * stride_pin
                    + offsets_in * stride_pik)
        start_n = tl.load(ind_ptrs)
        offsets_n = start_n[:, None] * B_SIZE + tl.arange(0, B_SIZE)[None, :]
        offsets_n = offsets_n.reshape(BLOCK_N)

        mask_p1 = offsets_n < p_ctx
        k_ptrs = (PK + cur_b * stride_pkb + cur_h * stride_pkh
                  + offsets_n[:, None] * stride_pkn
                  + offsets_d[None, :] * stride_pkd)
        v_ptrs = (PV + cur_b * stride_pkb + cur_h * stride_pkh
                  + offsets_n[:, None] * stride_pkn
                  + offsets_d[None, :] * stride_pkd)
        k = tl.load(k_ptrs, mask=mask_p1[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_p1[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32)
        scores = scores + tl.where(mask_p1[None, :], 0.0, -1.0e6)

        weights = tl.full([BLOCK_N], TOKEN_WEIGHT, tl.float32)
        is_last = offsets_n == (p_ctx - 1)
        weights = tl.where((BLOCK_LAST_WEIGHT > 0) & is_last, BLOCK_LAST_WEIGHT, weights)
        weights = tl.where(mask_p1, weights, 0.0)

        m_ij = tl.max(scores, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        exp_scores = tl.exp(scores - m_i_new[:, None])
        exp_scores_w = exp_scores * weights[None, :]
        l_ij = tl.sum(exp_scores_w, 1)
        l_i = l_i * tl.exp(m_i - m_i_new) + l_ij
        acc = acc * tl.exp(m_i - m_i_new)[:, None] + tl.dot(exp_scores_w.to(v.dtype), v)
        m_i = m_i_new

    for k_id in range(0, (p_ctx2 + BLOCK_N - 1) // BLOCK_N):
        offsets_n_2 = k_id * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_p2 = offsets_n_2 < p_ctx2

        k_ptrs = (PK2 + cur_b * stride_pk2b + cur_h * stride_pk2h
                  + offsets_n_2[:, None] * stride_pk2n
                  + offsets_d[None, :] * stride_pk2d)
        v_ptrs = (PV2 + cur_b * stride_pk2b + cur_h * stride_pk2h
                  + offsets_n_2[:, None] * stride_pk2n
                  + offsets_d[None, :] * stride_pk2d)
        k = tl.load(k_ptrs, mask=mask_p2[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_p2[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32)
        scores = scores + tl.where(mask_p2[None, :], 0.0, -1.0e6)

        weights = tl.full([BLOCK_N], TOKEN_WEIGHT_2, tl.float32)
        is_last = offsets_n_2 == (p_ctx2 - 1)
        weights = tl.where((BLOCK_LAST_WEIGHT_2 > 0) & is_last, BLOCK_LAST_WEIGHT_2, weights)
        weights = tl.where(mask_p2, weights, 0.0)

        m_ij = tl.max(scores, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        exp_scores = tl.exp(scores - m_i_new[:, None])
        exp_scores_w = exp_scores * weights[None, :]
        l_ij = tl.sum(exp_scores_w, 1)
        l_i = l_i * tl.exp(m_i - m_i_new) + l_ij
        acc = acc * tl.exp(m_i - m_i_new)[:, None] + tl.dot(exp_scores_w.to(v.dtype), v)
        m_i = m_i_new

    acc /= l_i[:, None]
    o_ptrs = (O + cur_b * stride_qb + cur_h * stride_qh
              + offsets_m[:, None] * stride_qm
              + offsets_d[None, :] * stride_qd)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offsets_m[:, None] < m_ctx)

    m_i += tl.math.log(l_i)
    m_ptrs = (M + cur_b * stride_mb + cur_h * stride_mh
              + offsets_m * stride_mm)
    tl.store(m_ptrs, m_i, mask=offsets_m < m_ctx)


def sparse_indices_attn_res_2_2_varlen_fwd(
    q, k, v, pk, pv, pk2, pv2, indices, p_indices, topk, topk_p,
    token_weight=1, token_weight_2=1, block_last_weight=0,
    block_last_weight_2=0, down_scale=4,
):
    b_size = down_scale * down_scale
    block_n = max(64, b_size)
    batch, n_head, m_ctx, d_head = q.shape
    _, _, n_ctx, _ = k.shape
    p_ctx = pk.shape[2]
    p_ctx2 = pk2.shape[2]
    scale = 1 / math.sqrt(d_head)

    m = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    grid = (triton.cdiv(m_ctx, b_size), batch * n_head)
    _sparse_indices_attn_res_2_2_varlen_fwd[grid](
        q, k, v, o, m, indices,
        pk, pv, p_indices,
        pk2, pv2,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        pk.stride(0), pk.stride(1), pk.stride(2), pk.stride(3),
        p_indices.stride(0), p_indices.stride(1), p_indices.stride(2), p_indices.stride(3),
        pk2.stride(0), pk2.stride(1), pk2.stride(2), pk2.stride(3),
        m_ctx, n_ctx, p_ctx, p_ctx2, n_head, scale, topk, topk_p,
        down_scale, token_weight, token_weight_2, block_last_weight, block_last_weight_2,
        b_size, block_n,
        D_HEAD=d_head,
    )

    return o, m


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_delta_b, stride_delta_h, stride_delta_m,
    n_ctx,
    NUM_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pid_bh = tl.program_id(1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    off_n = tl.arange(0, HEAD_DIM)
    mask_m = off_m < n_ctx

    o = tl.load(O + cur_b * stride_ob + cur_h * stride_oh
                + off_m[:, None] * stride_om + off_n[None, :] * stride_od,
                mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + cur_b * stride_dob + cur_h * stride_doh
                 + off_m[:, None] * stride_dom + off_n[None, :] * stride_dod,
                 mask=mask_m[:, None], other=0.0)
    delta = tl.sum(o * do, axis=-1)

    tl.store(Delta + cur_b * stride_delta_b + cur_h * stride_delta_h + off_m * stride_delta_m,
             delta, mask=mask_m)


@triton.jit
def _attn_bwd_res_dqkv(
    DQ, DK, DV,
    Q, K, V,
    PK, PV,
    PK2, PV2,
    DO, M, D,
    INDICES, OFFSETS, FLAT_INDICES,
    PREV_INDICES,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_pkb, stride_pkh, stride_pkn, stride_pkd,
    stride_pk2b, stride_pk2h, stride_pk2n, stride_pk2d,
    stride_ib, stride_ih, stride_im, stride_ik,
    stride_ob, stride_oh, stride_on,
    stride_fib, stride_fih, stride_fink,
    stride_pib, stride_pih, stride_pim, stride_pik,
    H, n_ctx, p_ctx, p_ctx2, scale, block_last_weight, block_last_weight_2,
    TOPK: tl.constexpr,
    TOPK_P: tl.constexpr,
    TOKEN_WEIGHT: tl.constexpr,
    TOKEN_WEIGHT_2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bhid = tl.program_id(1)
    cur_b = bhid // H
    cur_h = bhid % H

    adj_q = (stride_qb * cur_b + stride_qh * cur_h).to(tl.int64)
    adj_do = (stride_dob * cur_b + stride_doh * cur_h).to(tl.int64)
    adj_k = (stride_kb * cur_b + stride_kh * cur_h).to(tl.int64)
    adj_pk = (stride_pkb * cur_b + stride_pkh * cur_h).to(tl.int64)
    adj_pk2 = (stride_pk2b * cur_b + stride_pk2h * cur_h).to(tl.int64)
    pid = tl.program_id(0)
    off_chz = (bhid * n_ctx).to(tl.int64)

    Q += adj_q
    K += adj_k
    V += adj_k
    DO += adj_do
    DQ += adj_q
    DK += adj_k
    DV += adj_k
    M += off_chz
    D += off_chz

    PK += adj_pk
    PV += adj_pk
    PK2 += adj_pk2
    PV2 += adj_pk2

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < n_ctx
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    do = tl.load(DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dod,
                 mask=mask_m[:, None], other=0.0)
    Di = tl.load(D + offs_m, mask=mask_m, other=0.0)
    m = tl.load(M + offs_m, mask=mask_m, other=0.0)[:, None]

    for k_id in tl.static_range(TOPK):
        indices_ptrs = INDICES + cur_b * stride_ib + cur_h * stride_ih + pid * stride_im + k_id * stride_ik
        index = tl.load(indices_ptrs)
        offs_n = index * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_n = offs_n < n_ctx

        kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
        vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
        kT = tl.load(kT_ptrs, mask=mask_n[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_n[None, :], other=0.0)

        qk = tl.dot(q, kT).to(tl.float32)
        qk = qk + tl.where(mask_n[None, :], 0.0, -1.0e6)
        p = tl.math.exp(qk - m)

        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    prev_m_id = pid // BLOCK_M
    for k_id in tl.static_range(TOPK_P):
        indices_ptrs = PREV_INDICES + cur_b * stride_pib + cur_h * stride_pih + prev_m_id * stride_pim + k_id * stride_pik
        index = tl.load(indices_ptrs)
        offs_n = index * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_p1 = offs_n < p_ctx

        kT_ptrs = PK + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd
        vT_ptrs = PV + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd
        kT = tl.load(kT_ptrs, mask=mask_p1[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_p1[None, :], other=0.0)

        qk = tl.dot(q, kT).to(tl.float32)
        qk = qk + tl.where(mask_p1[None, :], 0.0, -1.0e6)

        weights = tl.full([BLOCK_M], TOKEN_WEIGHT, tl.float32)
        is_last = offs_n == (p_ctx - 1)
        weights = tl.where((block_last_weight > 0) & is_last, block_last_weight, weights)
        weights = tl.where(mask_p1, weights, 0.0)

        p = tl.math.exp(qk - m) * weights[None, :]
        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    for k_id in range(0, (p_ctx2 + BLOCK_M - 1) // BLOCK_M):
        offs_n_2 = k_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_p2 = offs_n_2 < p_ctx2

        kT_ptrs = PK2 + offs_n_2[None, :] * stride_pk2n + offs_k[:, None] * stride_pk2d
        vT_ptrs = PV2 + offs_n_2[None, :] * stride_pk2n + offs_k[:, None] * stride_pk2d
        kT = tl.load(kT_ptrs, mask=mask_p2[None, :], other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_p2[None, :], other=0.0)

        qk = tl.dot(q, kT).to(tl.float32)
        qk = qk + tl.where(mask_p2[None, :], 0.0, -1.0e6)

        weights = tl.full([BLOCK_M], TOKEN_WEIGHT_2, tl.float32)
        is_last = offs_n_2 == (p_ctx2 - 1)
        weights = tl.where((block_last_weight_2 > 0) & is_last, block_last_weight_2, weights)
        weights = tl.where(mask_p2, weights, 0.0)

        p = tl.math.exp(qk - m) * weights[None, :]
        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    dq_ptrs = DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    tl.store(dq_ptrs, dq, mask=mask_m[:, None])

    start_offset = tl.load(OFFSETS + cur_b * stride_ob + cur_h * stride_oh + pid * stride_on)
    end_offset = tl.load(OFFSETS + cur_b * stride_ob + cur_h * stride_oh + (pid + 1) * stride_on)

    offs_n = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = offs_n < n_ctx
    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
    vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
    kT = tl.load(kT_ptrs, mask=mask_n[None, :], other=0.0)
    vT = tl.load(vT_ptrs, mask=mask_n[None, :], other=0.0)

    dkT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dvT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    FLAT_INDICES += cur_b * stride_fib + cur_h * stride_fih
    for idx in range(start_offset, end_offset):
        q_id = tl.load(FLAT_INDICES + idx * stride_fink)
        offs_m_2 = q_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m_2 = offs_m_2 < n_ctx

        q_ptrs = Q + offs_m_2[:, None] * stride_qm + offs_k[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m_2[:, None], other=0.0)
        do = tl.load(DO + offs_m_2[:, None] * stride_dom + offs_k[None, :] * stride_dod,
                     mask=mask_m_2[:, None], other=0.0)
        Di = tl.load(D + offs_m_2, mask=mask_m_2, other=0.0)
        c_m = tl.load(M + offs_m_2, mask=mask_m_2, other=0.0)[:, None]

        qk = tl.dot(q, kT).to(tl.float32)
        qk = qk + tl.where(mask_n[None, :], 0.0, -1.0e6)
        p = tl.math.exp(qk - c_m)

        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])

        dvT_2 += tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
        dkT_2 += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    kv_offs = offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd
    kv_mask = mask_n[:, None]
    tl.store(DK + kv_offs, dkT_2, mask=kv_mask)
    tl.store(DV + kv_offs, dvT_2, mask=kv_mask)


@triton.jit
def _attn_bwd_res_dpkv1_sparse(
    DPK, DPV,
    Q,
    PK, PV,
    DO, M, D,
    OFFSETS, FLAT_INDICES,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_pkb, stride_pkh, stride_pkn, stride_pkd,
    stride_ob, stride_oh, stride_on,
    stride_fib, stride_fih, stride_fink,
    token_weight, block_last_weight,
    H, n_ctx, p_ctx, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bhid = tl.program_id(1)
    cur_b = bhid // H
    cur_h = bhid % H

    adj_q = (stride_qb * cur_b + stride_qh * cur_h).to(tl.int64)
    adj_do = (stride_dob * cur_b + stride_doh * cur_h).to(tl.int64)
    adj_pk = (stride_pkb * cur_b + stride_pkh * cur_h).to(tl.int64)
    pid = tl.program_id(0)
    off_chz = (bhid * n_ctx).to(tl.int64)

    Q += adj_q
    DO += adj_do
    M += off_chz
    D += off_chz

    PK += adj_pk
    PV += adj_pk
    DPK += adj_pk
    DPV += adj_pk

    offs_n = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_p = offs_n < p_ctx

    kT_ptrs = PK + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd
    vT_ptrs = PV + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd
    kT = tl.load(kT_ptrs, mask=mask_p[None, :], other=0.0)
    vT = tl.load(vT_ptrs, mask=mask_p[None, :], other=0.0)

    weights = tl.full([BLOCK_M], token_weight, tl.float32)
    is_last = offs_n == (p_ctx - 1)
    weights = tl.where((block_last_weight > 0) & is_last, block_last_weight, weights)
    weights = tl.where(mask_p, weights, 0.0)

    dkT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dvT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    start_offset = tl.load(OFFSETS + cur_b * stride_ob + cur_h * stride_oh + pid * stride_on)
    end_offset = tl.load(OFFSETS + cur_b * stride_ob + cur_h * stride_oh + (pid + 1) * stride_on)

    FLAT_INDICES += cur_b * stride_fib + cur_h * stride_fih
    for idx in range(start_offset, end_offset):
        pq_id = tl.load(FLAT_INDICES + idx * stride_fink)
        q_base = pq_id * BLOCK_M

        for q_off in range(0, BLOCK_M):
            q_id = q_base + q_off
            offs_m_2 = q_id * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m_2 < n_ctx

            q_ptrs = Q + offs_m_2[:, None] * stride_qm + offs_k[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(DO + offs_m_2[:, None] * stride_dom + offs_k[None, :] * stride_dod,
                         mask=mask_m[:, None], other=0.0)
            Di = tl.load(D + offs_m_2, mask=mask_m, other=0.0)
            c_m = tl.load(M + offs_m_2, mask=mask_m, other=0.0)[:, None]

            qk = tl.dot(q, kT).to(tl.float32)
            qk = qk + tl.where(mask_p[None, :], 0.0, -1.0e6)
            p = tl.math.exp(qk - c_m) * weights[None, :]

            dp = tl.dot(do, vT)
            ds = p * (dp - Di[:, None])

            dvT_2 += tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
            dkT_2 += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    kv_offs = offs_n[:, None] * stride_pkn + offs_k[None, :] * stride_pkd
    kv_mask = mask_p[:, None]
    tl.store(DPK + kv_offs, dkT_2, mask=kv_mask)
    tl.store(DPV + kv_offs, dvT_2, mask=kv_mask)


@triton.jit
def _attn_bwd_res_dpkv2(
    DPK, DPV,
    Q,
    PK, PV,
    DO, M, D,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_pkb, stride_pkh, stride_pkn, stride_pkd,
    token_weight, block_last_weight,
    H, n_ctx, p_ctx, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bhid = tl.program_id(1)
    cur_b = bhid // H
    cur_h = bhid % H

    adj_q = (stride_qb * cur_b + stride_qh * cur_h).to(tl.int64)
    adj_do = (stride_dob * cur_b + stride_doh * cur_h).to(tl.int64)
    adj_pk = (stride_pkb * cur_b + stride_pkh * cur_h).to(tl.int64)
    pid = tl.program_id(0)
    off_chz = (bhid * n_ctx).to(tl.int64)

    Q += adj_q
    DO += adj_do
    M += off_chz
    D += off_chz

    PK += adj_pk
    PV += adj_pk
    DPK += adj_pk
    DPV += adj_pk

    offs_n = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_p = offs_n < p_ctx

    kT_ptrs = PK + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd
    vT_ptrs = PV + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd
    kT = tl.load(kT_ptrs, mask=mask_p[None, :], other=0.0)
    vT = tl.load(vT_ptrs, mask=mask_p[None, :], other=0.0)

    weights = tl.full([BLOCK_M], token_weight, tl.float32)
    is_last = offs_n == (p_ctx - 1)
    weights = tl.where((block_last_weight > 0) & is_last, block_last_weight, weights)
    weights = tl.where(mask_p, weights, 0.0)

    dkT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dvT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for q_id in range(0, (n_ctx + BLOCK_M - 1) // BLOCK_M):
        offs_m_2 = q_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m_2 < n_ctx

        q_ptrs = Q + offs_m_2[:, None] * stride_qm + offs_k[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + offs_m_2[:, None] * stride_dom + offs_k[None, :] * stride_dod,
                     mask=mask_m[:, None], other=0.0)
        Di = tl.load(D + offs_m_2, mask=mask_m, other=0.0)
        c_m = tl.load(M + offs_m_2, mask=mask_m, other=0.0)[:, None]

        qk = tl.dot(q, kT).to(tl.float32)
        qk = qk + tl.where(mask_p[None, :], 0.0, -1.0e6)
        p = tl.math.exp(qk - c_m) * weights[None, :]

        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])

        dvT_2 += tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
        dkT_2 += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    kv_offs = offs_n[:, None] * stride_pkn + offs_k[None, :] * stride_pkd
    kv_mask = mask_p[:, None]
    tl.store(DPK + kv_offs, dkT_2, mask=kv_mask)
    tl.store(DPV + kv_offs, dvT_2, mask=kv_mask)


def sparse_indices_attn_res_scatter_2_2_bwd_v2_varlen(
    do, q, k, v, pk, pv, pk2, pv2, o, m, indices, prev_indices,
    topk, topk_p, block_size, token_weight, token_weight_2,
    block_last_weight, block_last_weight_2,
):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dpk = torch.empty_like(pk)
    dpv = torch.empty_like(pv)
    dpk2 = torch.empty_like(pk2)
    dpv2 = torch.empty_like(pv2)
    scale = 1 / math.sqrt(q.size(-1))

    p_ctx = pk.shape[2]
    p_ctx2 = pk2.shape[2]
    offsets, flat_indices = transpose_indices(indices, p_ctx)
    offsets_p, flat_indices_p = transpose_indices(prev_indices, p_ctx2)

    head_dim = q.shape[-1]
    batch, n_head, m_ctx = q.shape[:3]
    pre_block = 64
    block_n = 64
    arg_k = k * scale
    arg_pk = pk * scale
    arg_pk2 = pk2 * scale

    pre_grid = (triton.cdiv(m_ctx, pre_block), batch * n_head)
    delta = torch.empty_like(m)
    if not do.is_contiguous():
        do = do.contiguous()

    _attn_bwd_preprocess[pre_grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        m_ctx,
        NUM_HEAD=n_head,
        BLOCK_M=pre_block,
        HEAD_DIM=head_dim,
    )

    grid = (triton.cdiv(m_ctx, block_size), batch * n_head)
    _attn_bwd_res_dqkv[grid](
        dq, dk, dv,
        q, arg_k, v,
        arg_pk, pv,
        arg_pk2, pv2,
        do, m, delta,
        indices, offsets, flat_indices,
        prev_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        pk.stride(0), pk.stride(1), pk.stride(2), pk.stride(3),
        pk2.stride(0), pk2.stride(1), pk2.stride(2), pk2.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        offsets.stride(0), offsets.stride(1), offsets.stride(2),
        flat_indices.stride(0), flat_indices.stride(1), flat_indices.stride(2),
        prev_indices.stride(0), prev_indices.stride(1), prev_indices.stride(2), prev_indices.stride(3),
        n_head, m_ctx, p_ctx, p_ctx2, scale, block_last_weight, block_last_weight_2,
        topk, topk_p, token_weight, token_weight_2,
        block_size, block_n,
        HEAD_DIM=head_dim,
    )

    grid = (triton.cdiv(p_ctx, block_size), batch * n_head)
    _attn_bwd_res_dpkv1_sparse[grid](
        dpk, dpv,
        q,
        arg_pk, pv,
        do, m, delta,
        offsets_p, flat_indices_p,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        pk.stride(0), pk.stride(1), pk.stride(2), pk.stride(3),
        offsets_p.stride(0), offsets_p.stride(1), offsets_p.stride(2),
        flat_indices_p.stride(0), flat_indices_p.stride(1), flat_indices_p.stride(2),
        token_weight, block_last_weight,
        n_head, m_ctx, p_ctx, scale,
        block_size, block_n,
        HEAD_DIM=head_dim,
    )

    grid = (triton.cdiv(p_ctx2, block_size), batch * n_head)
    _attn_bwd_res_dpkv2[grid](
        dpk2, dpv2,
        q,
        arg_pk2, pv2,
        do, m, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        pk2.stride(0), pk2.stride(1), pk2.stride(2), pk2.stride(3),
        token_weight_2, block_last_weight_2,
        n_head, m_ctx, p_ctx2, scale,
        block_size, block_n,
        HEAD_DIM=head_dim,
    )

    return dq, dk, dv, None, dpk, dpv, None, dpk2, dpv2
