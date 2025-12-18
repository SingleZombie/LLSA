import math

import torch
import triton
import triton.language as tl
from .indices_transpose import transpose_indices


@triton.jit
def _sparse_indices_attn_res_2_2_fwd(
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
    m_ctx, n_ctx, NUM_HEAD: tl.constexpr,
    scale: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P: tl.constexpr,
    B_LENGTH: tl.constexpr,
    TOKEN_WEIGHT: tl.constexpr,
    TOKEN_WEIGHT_2: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    D_HEAD: tl.constexpr = 64,

):
    B_SIZE: tl.constexpr = B_LENGTH * B_LENGTH
    B_SIZE_2: tl.constexpr = B_SIZE * B_SIZE

    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    offsets_m = tl.arange(0, B_SIZE) + pid_m * B_SIZE

    NUM_M_BLOCKS: tl.constexpr = BLOCK_M // B_SIZE
    NUM_N_BLOCKS: tl.constexpr = BLOCK_N // B_SIZE

    offsets_d = tl.arange(0, D_HEAD)

    q_ptrs = (Q + cur_b * stride_qb + cur_h * stride_qh
              + offsets_m[:, None] * stride_qm
              + offsets_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offsets_m[:, None]
                < m_ctx, other=0.)
    q *= scale

    m_i = tl.full([B_LENGTH*B_LENGTH], -float('inf'), tl.float32)
    l_i = tl.zeros([B_LENGTH*B_LENGTH], tl.float32)
    acc = tl.zeros([B_LENGTH*B_LENGTH, D_HEAD], tl.float32)

    for k_id in range(0, TOPK // NUM_N_BLOCKS):
        offsets_in = k_id * NUM_N_BLOCKS + tl.arange(0, NUM_N_BLOCKS)
        ind_ptrs = (INDICES + cur_b * stride_ib + cur_h * stride_ih +
                    pid_m * stride_im +
                    offsets_in * stride_ik)
        start_n = tl.load(ind_ptrs)
        offsets_n = start_n[:, None] * B_SIZE + \
            tl.arange(0, B_SIZE)[None, :]
        offsets_n = offsets_n.reshape(BLOCK_N)

        k_ptrs = (K + cur_b * stride_kb + cur_h * stride_kh
                    + offsets_n[:, None] * stride_kn
                    + offsets_d[None, :] * stride_kd)
        v_ptrs = (V + cur_b * stride_vb + cur_h * stride_vh
                    + offsets_n[:, None] * stride_vn
                    + offsets_d[None, :] * stride_vd)

        attn_mask = offsets_n[None, :] < n_ctx
        k = tl.load(k_ptrs)

        v = tl.load(v_ptrs, mask=offsets_n[:, None]
                    < n_ctx, other=0.)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) + \
            tl.where(attn_mask, 0, -1.0e6)
        m_ij = tl.max(scores, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        exp_scores = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(exp_scores, 1)
        l_i = l_i * tl.exp(m_i - m_i_new) + l_ij
        acc = acc * tl.exp(m_i - m_i_new)[:, None] + \
            tl.dot(exp_scores.to(v.dtype), v)
        m_i = m_i_new

    prev_m_id = pid_m // B_SIZE

    for k_id in range(0, TOPK_P // NUM_N_BLOCKS):
        offsets_in = k_id * NUM_N_BLOCKS + tl.arange(0, NUM_N_BLOCKS)
        ind_ptrs = (PREV_INDICES + cur_b * stride_pib + cur_h * stride_pih +
                    prev_m_id * stride_pin +
                    offsets_in * stride_pik)
        start_n = tl.load(ind_ptrs)

        offsets_n = start_n[:, None] * B_SIZE + \
            tl.arange(0, B_SIZE)[None, :]
        offsets_n = offsets_n.reshape(BLOCK_N)

        k_ptrs = (PK + cur_b * stride_pkb + cur_h * stride_pkh
                  + offsets_n[:, None] * stride_pkn
                  + offsets_d[None, :] * stride_pkd)
        v_ptrs = (PV + cur_b * stride_pkb + cur_h * stride_pkh
                  + offsets_n[:, None] * stride_pkn
                  + offsets_d[None, :] * stride_pkd)

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32)
        m_ij = tl.max(scores, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        exp_scores = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(exp_scores, 1)
        l_i = l_i * tl.exp(m_i - m_i_new) + l_ij * TOKEN_WEIGHT
        acc = acc * tl.exp(m_i - m_i_new)[:, None] + \
            tl.dot(exp_scores.to(v.dtype), v) * TOKEN_WEIGHT
        m_i = m_i_new

    for k_id in range(0, n_ctx // B_SIZE_2 // BLOCK_N):

        offsets_n_2 = k_id * BLOCK_N + tl.arange(0, BLOCK_N)

        k_ptrs = (PK2 + cur_b * stride_pk2b + cur_h * stride_pk2h
                  + offsets_n_2[:, None] * stride_pk2n
                  + offsets_d[None, :] * stride_pk2d)
        v_ptrs = (PV2 + cur_b * stride_pk2b + cur_h * stride_pk2h
                  + offsets_n_2[:, None] * stride_pk2n
                  + offsets_d[None, :] * stride_pk2d)

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32)
        m_ij = tl.max(scores, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        exp_scores = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(exp_scores, 1)
        l_i = l_i * tl.exp(m_i - m_i_new) + l_ij * TOKEN_WEIGHT_2
        acc = acc * tl.exp(m_i - m_i_new)[:, None] + \
            tl.dot(exp_scores.to(v.dtype), v) * TOKEN_WEIGHT_2
        m_i = m_i_new

    acc /= l_i[:, None]
    o_ptrs = (O + cur_b * stride_qb + cur_h * stride_qh
              + offsets_m[:, None] * stride_qm
                + offsets_d[None, :] * stride_qd)

    tl.store(o_ptrs, acc.to(O.dtype.element_ty),
             mask=offsets_m[:, None] < m_ctx)

    m_i += tl.math.log(l_i)
    m_ptrs = (M + cur_b * stride_mb + cur_h * stride_mh
              + offsets_m * stride_mm)

    tl.store(m_ptrs, m_i, mask=offsets_m < n_ctx)


def sparse_indices_attn_res_2_2_fwd(q, k, v, pk, pv,
                                    pk2, pv2,
                                    indices, p_indices,
                                    topk, topk_p, token_weight=1, token_weight_2=1, down_scale=4):

    B_SIZE = down_scale * down_scale
    BLOCK_N = max(64, B_SIZE)
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    scale = 1 / math.sqrt(D)

    m = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    grid2 = (triton.cdiv(M, B_SIZE), B * H)
    _sparse_indices_attn_res_2_2_fwd[grid2](
        q, k, v, o, m, indices,
        pk, pv, p_indices,
        pk2, pv2,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        indices.stride(0), indices.stride(
            1), indices.stride(2), indices.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        pk.stride(0), pk.stride(1), pk.stride(2), pk.stride(3),
        p_indices.stride(0), p_indices.stride(
            1), p_indices.stride(2), p_indices.stride(3),
        pk2.stride(0), pk2.stride(1), pk2.stride(2), pk2.stride(3),
        M, N, H, scale, topk, topk_p,
        down_scale, token_weight, token_weight_2, B_SIZE, BLOCK_N,
        D_HEAD=D,
    )

    return o, m


@triton.jit
def _attn_bwd_preprocess(O, DO,
                         Delta,
                         stride_ob, stride_oh, stride_om, stride_od,
                         stride_dob, stride_doh, stride_dom, stride_dod,
                         stride_delata_b, stride_delta_h, stride_delta_m,
                         NUM_HEAD: tl.constexpr,
                         BLOCK_M: tl.constexpr,
                         HEAD_DIM: tl.constexpr
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pid_bh = tl.program_id(1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    off_n = tl.arange(0, HEAD_DIM)

    o = tl.load(O + cur_b * stride_ob + cur_h * stride_oh +
                off_m[:, None] * stride_om + off_n[None, :] * stride_od)
    do = tl.load(DO + cur_b * stride_dob + cur_h * stride_doh +
                 off_m[:, None] * stride_dom + off_n[None, :] * stride_dod)
    delta = tl.sum(o * do, axis=-1)

    tl.store(Delta + cur_b * stride_delata_b +
             cur_h * stride_delta_h + off_m * stride_delta_m, delta)


@triton.jit
def _attn_bwd_res_dqkv(DQ, DK, DV, DPK, DPV,
                       DPK2, DPV2,
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
                       H, n_ctx, scale,
                       TOPK: tl.constexpr,
                       TOPK_P: tl.constexpr,
                       TOKEN_WEIGHT: tl.constexpr,
                       TOKEN_WEIGHT_2: tl.constexpr,
                       BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr,
                       HEAD_DIM: tl.constexpr):
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
    DPK += adj_pk
    DPV += adj_pk

    PK2 += adj_pk2
    PV2 += adj_pk2
    DPK2 += adj_pk2
    DPV2 += adj_pk2

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    q = tl.load(q_ptrs)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    do = tl.load(DO + offs_m[:, None] * stride_dom +
                 offs_k[None, :] * stride_dod)

    Di = tl.load(D + offs_m)

    m = tl.load(M + offs_m)
    m = m[:, None]

    for k_id in tl.static_range(TOPK):
        indices_ptrs = INDICES + cur_b * stride_ib + cur_h * \
            stride_ih + pid * stride_im + k_id * stride_ik
        index = tl.load(indices_ptrs)
        offs_n = index * BLOCK_M + tl.arange(0, BLOCK_M)

        kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
        vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd

        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)

        qk = tl.dot(q, kT)
        p = tl.math.exp(qk - m)

        dp = tl.dot(do, vT)

        ds = p * (dp - Di[:, None])

        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    prev_m_id = pid // BLOCK_M

    for k_id in tl.static_range(TOPK_P):
        indices_ptrs = PREV_INDICES + cur_b * stride_pib + cur_h * \
            stride_pih + prev_m_id * stride_pim + k_id * stride_pik
        index = tl.load(indices_ptrs)
        offs_n = index * BLOCK_M + tl.arange(0, BLOCK_M)

        kT_ptrs = PK + offs_n[None, :] * \
            stride_pkn + offs_k[:, None] * stride_pkd
        vT_ptrs = PV + offs_n[None, :] * \
            stride_pkn + offs_k[:, None] * stride_pkd

        vT = tl.load(vT_ptrs)
        kT = tl.load(kT_ptrs)

        qk = tl.dot(q, kT)
        p = tl.math.exp(qk - m) * TOKEN_WEIGHT

        dp = tl.dot(do, vT)

        ds = p * (dp - Di[:, None])

        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    for k_id in range(n_ctx // BLOCK_M // BLOCK_M // BLOCK_M):
        offs_n_2 = k_id * BLOCK_M + tl.arange(0, BLOCK_M)

        kT_ptrs = PK2 + offs_n_2[None, :] * \
            stride_pk2n + offs_k[:, None] * stride_pk2d
        vT_ptrs = PV2 + offs_n_2[None, :] * \
            stride_pk2n + offs_k[:, None] * stride_pk2d

        vT = tl.load(vT_ptrs)
        kT = tl.load(kT_ptrs)

        qk = tl.dot(q, kT)
        p = tl.math.exp(qk - m) * TOKEN_WEIGHT_2

        dp = tl.dot(do, vT)

        ds = p * (dp - Di[:, None])

        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    dq_ptrs = DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    tl.store(dq_ptrs, dq)

    start_offset = tl.load(OFFSETS + cur_b * stride_ob +
                           cur_h * stride_oh + pid * stride_on)
    end_offset = tl.load(OFFSETS + cur_b * stride_ob +
                         cur_h * stride_oh + (pid + 1) * stride_on)

    offs_n = pid * BLOCK_M + tl.arange(0, BLOCK_M)

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
    vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd

    kT = tl.load(kT_ptrs)
    vT = tl.load(vT_ptrs)

    dkT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dvT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    FLAT_INDICES += cur_b * stride_fib + cur_h * stride_fih
    for id in range(start_offset, end_offset):
        q_id = tl.load(FLAT_INDICES + id * stride_fink)
        offs_m_2 = q_id * BLOCK_M + tl.arange(0, BLOCK_M)
        q_ptrs = Q + offs_m_2[:, None] * \
            stride_qm + offs_k[None, :] * stride_qd
        q = tl.load(q_ptrs)

        do = tl.load(DO + offs_m_2[:, None] * stride_dom +
                     offs_k[None, :] * stride_dod)

        Di = tl.load(D + offs_m_2)

        c_m = tl.load(M + offs_m_2)
        c_m = c_m[:, None]

        qk = tl.dot(q, kT).to(tl.float32)
        p = tl.math.exp(qk - c_m)

        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])

        dvT_2 += tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
        dkT_2 += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    kv_offs = offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd

    tl.store(DK + kv_offs, dkT_2)
    tl.store(DV + kv_offs, dvT_2)


@triton.jit
def _attn_bwd_res_dpkv2(DPK, DPV,
                        Q,
                        PK, PV,
                        DO, M, D,
                        stride_qb, stride_qh, stride_qm, stride_qd,
                        stride_dob, stride_doh, stride_dom, stride_dod,
                        stride_pkb, stride_pkh, stride_pkn, stride_pkd,
                        token_weight,
                        H, n_ctx, scale,
                        BLOCK_M: tl.constexpr,
                        BLOCK_N: tl.constexpr,
                        HEAD_DIM: tl.constexpr):
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

    kT_ptrs = PK + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd
    vT_ptrs = PV + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd

    kT = tl.load(kT_ptrs)
    vT = tl.load(vT_ptrs)

    dkT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dvT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for q_id in range(n_ctx // BLOCK_M):
        offs_m_2 = q_id * BLOCK_M + tl.arange(0, BLOCK_M)
        q_ptrs = Q + offs_m_2[:, None] * \
            stride_qm + offs_k[None, :] * stride_qd
        q = tl.load(q_ptrs)

        do = tl.load(DO + offs_m_2[:, None] * stride_dom +
                     offs_k[None, :] * stride_dod)

        Di = tl.load(D + offs_m_2)

        c_m = tl.load(M + offs_m_2)
        c_m = c_m[:, None]

        qk = tl.dot(q, kT).to(tl.float32)
        p = tl.math.exp(qk - c_m) * token_weight

        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])

        dvT_2 += tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
        dkT_2 += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    kv_offs = offs_n[:, None] * stride_pkn + offs_k[None, :] * stride_pkd

    tl.store(DPK + kv_offs, dkT_2)
    tl.store(DPV + kv_offs, dvT_2)


@triton.jit
def _attn_bwd_res_dpkv1_sparse(DPK, DPV,
                               Q,
                               PK, PV,
                               DO, M, D,
                               OFFSETS, FLAT_INDICES,
                               stride_qb, stride_qh, stride_qm, stride_qd,
                               stride_dob, stride_doh, stride_dom, stride_dod,
                               stride_pkb, stride_pkh, stride_pkn, stride_pkd,
                               stride_ob, stride_oh, stride_on,
                               stride_fib, stride_fih, stride_fink,
                               token_weight,
                               H, n_ctx, scale,
                               BLOCK_M: tl.constexpr,
                               BLOCK_N: tl.constexpr,
                               HEAD_DIM: tl.constexpr):
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

    kT_ptrs = PK + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd
    vT_ptrs = PV + offs_n[None, :] * stride_pkn + offs_k[:, None] * stride_pkd

    kT = tl.load(kT_ptrs)
    vT = tl.load(vT_ptrs)

    dkT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dvT_2 = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    start_offset = tl.load(OFFSETS + cur_b * stride_ob +
                           cur_h * stride_oh + pid * stride_on)
    end_offset = tl.load(OFFSETS + cur_b * stride_ob +
                         cur_h * stride_oh + (pid + 1) * stride_on)

    FLAT_INDICES += cur_b * stride_fib + cur_h * stride_fih

    for id in range(start_offset, end_offset):
        pq_id = tl.load(FLAT_INDICES + id * stride_fink)
        for q_id in range(pq_id * BLOCK_M, (pq_id+1) * BLOCK_M):
            offs_m_2 = q_id * BLOCK_M + tl.arange(0, BLOCK_M)
            q_ptrs = Q + offs_m_2[:, None] * \
                stride_qm + offs_k[None, :] * stride_qd
            q = tl.load(q_ptrs)

            do = tl.load(DO + offs_m_2[:, None] * stride_dom +
                         offs_k[None, :] * stride_dod)

            Di = tl.load(D + offs_m_2)

            c_m = tl.load(M + offs_m_2)
            c_m = c_m[:, None]

            qk = tl.dot(q, kT).to(tl.float32)
            p = tl.math.exp(qk - c_m) * token_weight

            dp = tl.dot(do, vT)
            ds = p * (dp - Di[:, None])

            dvT_2 += tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
            dkT_2 += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    kv_offs = offs_n[:, None] * stride_pkn + offs_k[None, :] * stride_pkd

    tl.store(DPK + kv_offs, dkT_2)
    tl.store(DPV + kv_offs, dvT_2)


def sparse_indices_attn_res_scatter_2_2_bwd_v2(do, q, k, v,
                                               pk, pv,
                                               pk2, pv2,
                                               o, m, indices,
                                               prev_indices, topk, topk_p,
                                               block_size, token_weight, token_weight_2):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dpk = torch.empty_like(pk)
    dpv = torch.empty_like(pv)
    dpk2 = torch.empty_like(pk2)
    dpv2 = torch.empty_like(pv2)
    scale = 1 / math.sqrt(q.size(-1))

    offsets, flat_indices = transpose_indices(indices)
    offsets_p, flat_indices_p = transpose_indices(prev_indices)

    HEAD_DIM = q.shape[-1]

    BATCH, N_HEAD, M_CTX = q.shape[:3]
    PRE_BLOCK = min(64, M_CTX)
    BLOCK_N = 64
    arg_k = k
    arg_k = arg_k * scale
    arg_pk = pk * scale
    arg_pk2 = pk2 * scale
    assert M_CTX % PRE_BLOCK == 0

    pre_grid = (M_CTX // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(m)
    if not do.is_contiguous():
        do = do.contiguous()

    _attn_bwd_preprocess[pre_grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2), NUM_HEAD=N_HEAD,
        BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM
    )

    grid = (M_CTX // block_size, BATCH * N_HEAD)

    _attn_bwd_res_dqkv[grid](
        dq, dk, dv, dpk, dpv,
        dpk2, dpv2,
        q, arg_k, v,
        arg_pk, pv,
        arg_pk2, pv2,
        do, m, delta, indices, offsets, flat_indices,
        prev_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        pk.stride(0), pk.stride(1), pk.stride(2), pk.stride(3),
        pk2.stride(0), pk2.stride(1), pk2.stride(2), pk2.stride(3),
        indices.stride(0), indices.stride(
            1), indices.stride(2), indices.stride(3),
        offsets.stride(0), offsets.stride(1), offsets.stride(2),
        flat_indices.stride(0), flat_indices.stride(1), flat_indices.stride(2),
        prev_indices.stride(0), prev_indices.stride(
            1), prev_indices.stride(2), prev_indices.stride(3),
        N_HEAD, M_CTX, scale,
        topk, topk_p, token_weight, token_weight_2,
        block_size, BLOCK_N,
        HEAD_DIM=HEAD_DIM,
    )

    grid = (M_CTX // block_size // block_size, BATCH * N_HEAD)

    _attn_bwd_res_dpkv1_sparse[grid](
        dpk, dpv, q,
        arg_pk, pv,
        do, m, delta,
        offsets_p, flat_indices_p,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        pk.stride(0), pk.stride(1), pk.stride(2), pk.stride(3),
        offsets_p.stride(0), offsets_p.stride(1), offsets_p.stride(2),
        flat_indices_p.stride(0), flat_indices_p.stride(
            1), flat_indices_p.stride(2),
        token_weight,
        N_HEAD, M_CTX, scale,
        block_size, BLOCK_N,
        HEAD_DIM=HEAD_DIM,
    )

    grid = (M_CTX // block_size // block_size // block_size, BATCH * N_HEAD)

    _attn_bwd_res_dpkv2[grid](
        dpk2, dpv2, q,
        arg_pk2, pv2,
        do, m, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        pk2.stride(0), pk2.stride(1), pk2.stride(2), pk2.stride(3),
        token_weight_2,
        N_HEAD, M_CTX, scale,
        block_size, BLOCK_N,
        HEAD_DIM=HEAD_DIM,
    )

    return dq, dk, dv, None, dpk, dpv, None, dpk2, dpv2
