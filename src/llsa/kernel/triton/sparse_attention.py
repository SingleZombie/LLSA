import math
import time
from typing import Tuple

from einops import rearrange
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.standard import _log2, sum, zeros_like
from .indices_transpose import transpose_indices


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = tl.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape)
    right_idx = tl.reshape(right_idx, x.shape)

    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth,
                                   signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    cond = cond.to(tl.int1)

    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))

    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr,
                   n_dims: tl.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [
            n_outer * 2**(n_dims - 1 - stage), 2, 2**stage
        ]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape),
            x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x,
            ids,
            dim: tl.constexpr = None,
            descending: tl.constexpr = False):
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(_dim == len(x.shape) - 1,
                     "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = _log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending,
                                n_dims)
    return x, ids


@triton.jit
def _compute_sparse_indices_fwd(PQ, PK, INDICES,
                                stride_pqb, stride_pqh, stride_pqm, stride_pqd,
                                stride_pkb, stride_pkh, stride_pkn, stride_pkd,
                                stride_ib, stride_ih, stride_im, stride_ik,
                                m_ctx, n_ctx, NUM_HEAD: tl.constexpr,
                                topk: tl.constexpr,
                                BLOCK_M: tl.constexpr,
                                BLOCK_N: tl.constexpr,
                                D_HEAD: tl.constexpr = 64):

    p_i_id = tl.program_id(0)
    p_bh = tl.program_id(1)
    cur_b = p_bh // NUM_HEAD
    cur_h = p_bh % NUM_HEAD

    offsets_m = p_i_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, D_HEAD)

    pq_ptrs = (PQ + cur_b * stride_pqb + cur_h * stride_pqh
               + offsets_m[:, None] * stride_pqm
               + offsets_d[None, :] * stride_pqd)

    pq = tl.load(pq_ptrs, mask=offsets_m[:, None]
                 < m_ctx, other=0.)

    b_i = tl.full([BLOCK_M, BLOCK_N], -1, dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    m_i = tl.arange(0, BLOCK_N) < BLOCK_N//2
    m_i = tl.broadcast_to(m_i[None, :], (BLOCK_M, BLOCK_N))
    for i_c in range(0, (n_ctx - 1) // BLOCK_N + 1):

        offsets_n = i_c * BLOCK_N + tl.arange(0, BLOCK_N)
        pkT_ptrs = (PK + cur_b * stride_pkb + cur_h * stride_pkh
                    + offsets_n[None, :] * stride_pkn
                    + offsets_d[:, None] * stride_pkd)

        pkT = tl.load(pkT_ptrs, mask=offsets_n[None, :]
                      < n_ctx, other=float('-inf'))
        scores = tl.dot(pq, pkT)
        b_i, b_ip = scores, b_i
        o_i, o_ip = tl.broadcast_to(tl.where(offsets_n < n_ctx, offsets_n, 0)[
                                    None, :], (BLOCK_M, BLOCK_N)), o_i

        n_dims: tl.constexpr = tl.standard._log2(BLOCK_N)
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

        if i_c != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(
                tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(
                b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims)
        else:
            b_i, o_i = _bitonic_merge(
                b_i, o_i.to(tl.int32), n_dims, True, n_dims)

    mask_top = tl.arange(0, BLOCK_N // topk) == 0
    indices_top = tl.sum(
        mask_top[None, :, None] * tl.reshape(o_i, [BLOCK_M, BLOCK_N // topk, topk]), 1)

    offsets_topk = tl.arange(0, topk)
    indices_ptrs = (INDICES + cur_b * stride_ib + cur_h * stride_ih
                    + offsets_m[:, None] * stride_im
                    + offsets_topk[None, :] * stride_ik)

    tl.store(indices_ptrs, indices_top, mask=offsets_m[:, None] < n_ctx)


def compute_indices(pq, pk, topk):
    B, H, M, D = pq.shape  # multi‑head format (B,H,L,D)
    _, _, N, _ = pk.shape  # multi‑head format (B,H,L,D)

    indices = torch.empty((B, H, M, topk), device=pq.device, dtype=torch.long)

    BLOCK_M = 64
    BLOCK_N = min(64, N)
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    _compute_sparse_indices_fwd[grid](
        pq, pk, indices,
        pq.stride(0), pq.stride(1), pq.stride(2), pq.stride(3),
        pk.stride(0), pk.stride(1), pk.stride(2), pk.stride(3),
        indices.stride(0), indices.stride(
            1), indices.stride(2), indices.stride(3),
        M, N, H,
        topk,
        BLOCK_M,
        BLOCK_N,
        D
    )
    return indices


@triton.jit
def _compute_sparse_indices_2_fwd(PQ, PK, PREV_INDICES, INDICES,
                                  stride_pqb, stride_pqh, stride_pqm, stride_pqd,
                                  stride_pkb, stride_pkh, stride_pkn, stride_pkd,
                                  stride_ib, stride_ih, stride_im, stride_ik,
                                  stride_ipb, stride_iph, stride_ipm, stride_ipk,
                                  m_ctx, n_ctx, NUM_HEAD: tl.constexpr,
                                  NUM_INDICES: tl.constexpr,
                                  P_BLOCK_SIZE: tl.constexpr,
                                  topk: tl.constexpr,
                                  BLOCK_M: tl.constexpr,
                                  BLOCK_N: tl.constexpr,
                                  D_HEAD: tl.constexpr = 64):

    p_i_id = tl.program_id(0)
    p_bh = tl.program_id(1)
    cur_b = p_bh // NUM_HEAD
    cur_h = p_bh % NUM_HEAD

    # offsets_m = p_i_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_m = p_i_id * P_BLOCK_SIZE + tl.arange(0, P_BLOCK_SIZE)
    offsets_d = tl.arange(0, D_HEAD)

    pq_ptrs = (PQ + cur_b * stride_pqb + cur_h * stride_pqh
               + offsets_m[:, None] * stride_pqm
               + offsets_d[None, :] * stride_pqd)

    pq = tl.load(pq_ptrs, mask=offsets_m[:, None]
                 < m_ctx, other=0.)

    b_i = tl.full([BLOCK_M, BLOCK_N], -1, dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
    m_i = tl.arange(0, BLOCK_N) < BLOCK_N//2
    m_i = tl.broadcast_to(m_i[None, :], (BLOCK_M, BLOCK_N))

    for loop_i in range(0, NUM_INDICES):
        prev_indices_ptrs = (PREV_INDICES + cur_b * stride_ipb + cur_h * stride_iph +
                             p_i_id * stride_ipm + loop_i * stride_ipk)
        prev_index = tl.load(prev_indices_ptrs)

        i_c = prev_index
        offsets_n = i_c * BLOCK_N + tl.arange(0, BLOCK_N)
        pkT_ptrs = (PK + cur_b * stride_pkb + cur_h * stride_pkh
                    + offsets_n[None, :] * stride_pkn
                    + offsets_d[:, None] * stride_pkd)

        pkT = tl.load(pkT_ptrs, mask=offsets_n[None, :]
                      < n_ctx, other=float('-inf'))
        scores = tl.dot(pq, pkT)
        b_i, b_ip = scores, b_i
        o_i, o_ip = tl.broadcast_to(tl.where(offsets_n < n_ctx, offsets_n, 0)[
            None, :], (BLOCK_M, BLOCK_N)), o_i

        n_dims: tl.constexpr = tl.standard._log2(BLOCK_N)
        for i in tl.static_range(1, n_dims):
            b_i, o_i = _bitonic_merge(
                b_i, o_i.to(tl.int32), i, 2, n_dims)

        if loop_i != 0:
            b_i, o_i = _bitonic_merge(b_i, o_i.to(
                tl.int32), n_dims, False, n_dims)
            b_i_new = b_ip * m_i + b_i * (1 - m_i)
            o_i_new = o_ip * m_i + o_i * (1 - m_i)
            b_i, o_i = _bitonic_merge(
                b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims)
        else:
            b_i, o_i = _bitonic_merge(
                b_i, o_i.to(tl.int32), n_dims, True, n_dims)

    mask_top = tl.arange(0, BLOCK_N // topk) == 0
    indices_top = tl.sum(
        mask_top[None, :, None] * tl.reshape(o_i, [BLOCK_M, BLOCK_N // topk, topk]), 1)

    offsets_topk = tl.arange(0, topk)
    indices_ptrs = (INDICES + cur_b * stride_ib + cur_h * stride_ih
                    + offsets_m[:, None] * stride_im
                    + offsets_topk[None, :] * stride_ik)

    tl.store(indices_ptrs, indices_top, mask=offsets_m[:, None] < n_ctx)


def compute_indices_2(pq, pk, topk, p_indices, p_block_size):
    B, H, M, D = pq.shape  # multi‑head format (B,H,L,D)
    _, _, N, _ = pk.shape  # multi‑head format (B,H,L,D)

    indices = torch.empty((B, H, M, topk), device=pq.device, dtype=torch.long)

    BLOCK_M = 16
    BLOCK_N = min(64, N)
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    _compute_sparse_indices_2_fwd[grid](
        pq, pk, p_indices, indices,
        pq.stride(0), pq.stride(1), pq.stride(2), pq.stride(3),
        pk.stride(0), pk.stride(1), pk.stride(2), pk.stride(3),
        indices.stride(0), indices.stride(
            1), indices.stride(2), indices.stride(3),
        p_indices.stride(0), p_indices.stride(
            1), p_indices.stride(2), p_indices.stride(3),
        M, N, H, p_indices.shape[-1], p_block_size,
        topk,
        p_block_size,
        p_block_size,
        D
    )
    return indices


@triton.jit
def _sparse_indices_attn_fwd(
    Q, K, V, O, M, INDICES,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_im, stride_ik,
    stride_mb, stride_mh, stride_mm,
    m_ctx, n_ctx, NUM_HEAD: tl.constexpr,
    scale: tl.constexpr,
    TOPK: tl.constexpr,
    P_LENGTH: tl.constexpr,
    B_LENGTH: tl.constexpr,
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,  # query rows per CTA
    # head dimension (must be <= 64 for bank‑conflict‑free dot)
    D_HEAD: tl.constexpr = 64,
):
    pid_m = tl.program_id(axis=0)  # row block id
    pid_bh = tl.program_id(axis=1)  # batch*head id
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    # p_row = pid_m // P_LENGTH
    # p_col = pid_m % P_LENGTH
    # new_q_id = (B_LENGTH * P_LENGTH * (B_LENGTH * p_row + tl.arange(0, B_LENGTH)))[:, None] + \
    #     (B_LENGTH * p_col + tl.arange(0, B_LENGTH))[None, :]

    # offsets_m = new_q_id.reshape(B_LENGTH*B_LENGTH)

    B_SIZE: tl.constexpr = B_LENGTH * B_LENGTH
    offsets_m = tl.arange(0, B_SIZE) + pid_m * B_SIZE

    # offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    NUM_M_BLOCKS: tl.constexpr = BLOCK_M // B_SIZE
    NUM_N_BLOCKS: tl.constexpr = BLOCK_N // B_SIZE

    offsets_d = tl.arange(0, D_HEAD)

    # load Q –  (M,D)
    q_ptrs = (Q + cur_b * stride_qb + cur_h * stride_qh
              + offsets_m[:, None] * stride_qm
              + offsets_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offsets_m[:, None]
                < m_ctx, other=0.)
    q *= scale

    # initialise softmax accumulators
    m_i = tl.full([B_SIZE], -float('inf'), tl.float32)
    l_i = tl.zeros([B_SIZE], tl.float32)
    acc = tl.zeros([B_SIZE, D_HEAD], tl.float32)

    # for m_id in range(0, NUM_M_BLOCKS):

    # for k_id in tl.static_range(0, TOPK // NUM_N_BLOCKS):
    #     offsets_in = k_id * NUM_N_BLOCKS + tl.arange(0, NUM_N_BLOCKS)
    #     ind_ptrs = (INDICES + cur_b * stride_ib + cur_h * stride_ih +
    #                 pid_m * stride_im +
    #                 offsets_in * stride_ik)
    #     start_n = tl.load(ind_ptrs)

    #     offsets_n = start_n[:, None] * B_LENGTH * B_LENGTH + \
    #         tl.arange(0, B_LENGTH * B_LENGTH)[None, :]
    #     offsets_n = offsets_n.reshape(BLOCK_N)

    for k_id in tl.static_range(0, TOPK):
        ind_ptrs = (INDICES + cur_b * stride_ib + cur_h * stride_ih +
                    pid_m * stride_im +
                    k_id * stride_ik)
        start_n = tl.load(ind_ptrs)

        offsets_n = start_n * B_SIZE + tl.arange(0, B_SIZE)

        # offsets_n = k_id * BLOCK_N * tl.arange(0, BLOCK_N)

        k_ptrs = (K + cur_b * stride_kb + cur_h * stride_kh
                    + offsets_n[:, None] * stride_kn
                    + offsets_d[None, :] * stride_kd)
        v_ptrs = (V + cur_b * stride_vb + cur_h * stride_vh
                    + offsets_n[:, None] * stride_vn
                    + offsets_d[None, :] * stride_vd)
        k = tl.load(k_ptrs, mask=offsets_n[:, None]
                    < n_ctx, other=-float('inf'))
        v = tl.load(v_ptrs, mask=offsets_n[:, None]
                    < n_ctx, other=0.)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32)
        m_ij = tl.max(scores, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        exp_scores = tl.exp(scores - m_i_new[:, None])
        l_ij = tl.sum(exp_scores, 1)
        l_i = l_i * tl.exp(m_i - m_i_new) + l_ij
        acc = acc * tl.exp(m_i - m_i_new)[:, None] + \
            tl.dot(exp_scores.to(v.dtype), v)
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


def sparse_indices_attn_fwd(q, k, v, indices, topk, down_scale=4):
    BLOCK_N = 64
    DOWN_SQ = down_scale * down_scale
    B, H, M, D = q.shape  # multi‑head format (B,H,L,D)
    _, _, N, _ = k.shape  # multi‑head format (B,H,L,D)
    scale = 1 / math.sqrt(D)
    SIZE = int((M ** 0.5) // down_scale)

    m = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    grid2 = (triton.cdiv(M, DOWN_SQ), B * H)
    _sparse_indices_attn_fwd[grid2](
        q, k, v, o, m, indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        indices.stride(0), indices.stride(
            1), indices.stride(2), indices.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        M, N, H, scale, topk,
        SIZE, down_scale, down_scale*down_scale, BLOCK_N,
        D_HEAD=D,
        # num_warps=4
    )

    return o, m


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         stride_ob, stride_oh, stride_om, stride_od,
                         stride_dob, stride_doh, stride_dom, stride_dod,
                         stride_delata_b, stride_delta_h, stride_delta_m,
                         NUM_HEAD: tl.constexpr,
                         BLOCK_M: tl.constexpr,
                         HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    pid_bh = tl.program_id(1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + cur_b * stride_ob + cur_h * stride_oh +
                off_m[:, None] * stride_om + off_n[None, :] * stride_od)
    do = tl.load(DO + cur_b * stride_dob + cur_h * stride_doh +
                 off_m[:, None] * stride_dom + off_n[None, :] * stride_dod)
    delta = tl.sum(o * do, axis=-1)
    # write-back
    tl.store(Delta + cur_b * stride_delata_b +
             cur_h * stride_delta_h + off_m * stride_delta_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(DK, DV,  #
                   Q, K, V,  #
                   DO,  #
                   M, D,  #
                   CROW_IND, COL_IND,
                   stride_qh, stride_qm, stride_qd,  #
                   stride_kh, stride_kn, stride_kd,
                   stride_crih, stride_crin,
                   stride_coih, stride_coin,
                   BLOCK_M: tl.constexpr,  #
                   BLOCK_N: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   N_CTX,
                   scale):
    bhid = tl.program_id(1)
    adj_q = (stride_qh * bhid).to(tl.int64)
    adj_k = (stride_kh * bhid).to(tl.int64)
    pid = tl.program_id(0)
    off_chz = (bhid * N_CTX).to(tl.int64)

    Q += adj_q
    K += adj_k
    V += adj_k
    DO += adj_q
    DK += adj_k
    DV += adj_k
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)

    crow_ptrs = CROW_IND + bhid * stride_crih + pid * stride_crin
    start_ind = tl.load(crow_ptrs)
    end_ind = tl.load(crow_ptrs+stride_crin)

    COL_IND += bhid * stride_coih

    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.load(K + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd)
    v = tl.load(V + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd)

    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    for i_idx in range(start_ind, end_ind):
        row_ind = tl.load(COL_IND + i_idx * stride_coin)
        row_ind = row_ind * BLOCK_M + tl.arange(0, BLOCK_M)

        qT_ptrs = Q + row_ind[None, :] * \
            stride_qm + offs_k[:, None] * stride_qd
        do_ptrs = DO + row_ind[:, None] * \
            stride_qm + offs_k[None, :] * stride_qd
        qT = tl.load(qT_ptrs)

        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + row_ind)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp(qkT - m[None, :])
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + row_ind)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))

    dv_ptrs = DV + offs_n[:, None] * \
        stride_kn + offs_k[None, :] * stride_kd
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= scale
    dk_ptrs = DK + offs_n[:, None] * \
        stride_kn + offs_k[None, :] * stride_kd
    tl.store(dk_ptrs, dk)

# the main inner-loop logic for computing dQ


@triton.jit
def _attn_bwd_dq(DQ, Q, K, V,  #
                 DO, M, D, INDICES,
                 stride_qh, stride_qm, stride_qd,  #
                 stride_kh, stride_kn, stride_kd,  #
                 H, N_CTX,  #
                 TOP_K: tl.constexpr,
                 BLOCK_M: tl.constexpr,  #
                 BLOCK_N: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr):
    bhid = tl.program_id(1)
    adj_q = (stride_qh * bhid).to(tl.int64)
    adj_k = (stride_kh * bhid).to(tl.int64)
    pid = tl.program_id(0)
    off_chz = (bhid * N_CTX).to(tl.int64)

    Q += adj_q
    K += adj_k
    V += adj_k
    DO += adj_q
    DQ += adj_q
    M += off_chz
    D += off_chz

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    q = tl.load(q_ptrs)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_qm +
                 offs_k[None, :] * stride_qd)

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)

    m = tl.load(M + offs_m)
    m = m[:, None]

    for blk_idx in range(TOP_K):
        indices_ptrs = INDICES + bhid * BLOCK_M * TOP_K + pid * TOP_K + blk_idx
        index = tl.load(indices_ptrs)
        offs_n = index * BLOCK_N + tl.arange(0, BLOCK_N)
        kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
        vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp(qk - m)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))

    dq_ptrs = DQ + offs_m[:, None] * stride_kn + offs_k[None, :] * stride_kd
    tl.store(dq_ptrs, dq)


@triton.jit
def _attn_bwd_dqkv_scatter(DQ, DK, DV,
                           Q, K, V,  #
                           DO, M, D, INDICES,
                           stride_qb, stride_qh, stride_qm, stride_qd,  #
                           stride_kb, stride_kh, stride_kn, stride_kd,  #
                           stride_dob, stride_doh, stride_dom, stride_dod,  #
                           stride_ib, stride_ih, stride_im, stride_ik,
                           H, N_CTX, scale,
                           TOP_K: tl.constexpr,
                           BLOCK_M: tl.constexpr,  #
                           BLOCK_N: tl.constexpr,  #
                           HEAD_DIM: tl.constexpr):
    bhid = tl.program_id(1)
    cur_b = bhid // H
    cur_h = bhid % H

    adj_q = (stride_qb * cur_b + stride_qh * cur_h).to(tl.int64)
    adj_do = (stride_dob * cur_b + stride_doh * cur_h).to(tl.int64)
    adj_k = (stride_kb * cur_b + stride_kh * cur_h).to(tl.int64)
    pid = tl.program_id(0)
    off_chz = (bhid * N_CTX).to(tl.int64)

    Q += adj_q
    K += adj_k
    V += adj_k
    DO += adj_do
    DQ += adj_q
    DK += adj_k
    DV += adj_k
    M += off_chz
    D += off_chz

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    q = tl.load(q_ptrs)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_dom +
                 offs_k[None, :] * stride_dod)

    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)

    m = tl.load(M + offs_m)
    m = m[:, None]

    NUM_N_BLOCKS: tl.constexpr = BLOCK_N // BLOCK_M

    for k_id in tl.static_range(TOP_K):
        indices_ptrs = INDICES + cur_b * stride_ib + \
            cur_h * stride_ih + pid * TOP_K + k_id
        index = tl.load(indices_ptrs)
        offs_n = index * BLOCK_N + tl.arange(0, BLOCK_N)

    # for k_id in tl.static_range(0, TOP_K // NUM_N_BLOCKS):
    #     offsets_in = k_id * NUM_N_BLOCKS + tl.arange(0, NUM_N_BLOCKS)
    #     indices_ptrs = (INDICES + cur_b * stride_ib + cur_h * stride_ih +
    #                     pid * stride_im + offsets_in * stride_ik)
    #     index = tl.load(indices_ptrs)

    #     offs_n = index[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
    #     offs_n = offs_n.reshape(BLOCK_N)

        kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
        vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd

        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)

        qk = tl.dot(q, kT).to(tl.float32)
        p = tl.math.exp(qk - m)
        # Compute dP and dS.
        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

        dvT = tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
        dkT = tl.dot(tl.trans(ds).to(q.dtype), q) * scale
        # dvT = tl.dot(tl.trans(p), do)
        # dkT = tl.dot(tl.trans(ds), q.to(tl.float32)) * scale

        kv_offs = offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd

        tl.atomic_add(DK + kv_offs, dkT.to(tl.float32))
        tl.atomic_add(DV + kv_offs, dvT.to(tl.float32))

    dq_ptrs = DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    tl.store(dq_ptrs, dq)


# @triton.jit
# def _attn_bwd_dqkv_scatter_group(DQ, DK, DV,
#                                  Q, K, V,  #
#                                  DO, M, D, INDICES,
#                                  stride_dkg, stride_dkb, stride_dkh, stride_dkm, stride_dkd,
#                                  stride_qb, stride_qh, stride_qm, stride_qd,  #
#                                  stride_kb, stride_kh, stride_kn, stride_kd,  #
#                                  stride_dob, stride_doh, stride_dom, stride_dod,  #
#                                  stride_ib, stride_ih, stride_im, stride_ik,
#                                  H, N_CTX, scale,
#                                  TOP_K: tl.constexpr,
#                                  BLOCK_M: tl.constexpr,  #
#                                  BLOCK_N: tl.constexpr,  #
#                                  HEAD_DIM: tl.constexpr,
#                                  NUM_GROUPS: tl.constexpr):
#     bhid = tl.program_id(1)
#     cur_b = bhid // H
#     cur_h = bhid % H

#     pid = tl.program_id(0)
#     cur_g = pid % NUM_GROUPS

#     adj_q = (stride_qb * cur_b + stride_qh * cur_h).to(tl.int64)
#     adj_do = (stride_dob * cur_b + stride_doh * cur_h).to(tl.int64)
#     adj_k = (stride_kb * cur_b + stride_kh * cur_h).to(tl.int64)
#     adj_dk = (stride_dkg * cur_g + stride_dkb *
#               cur_b + stride_dkh * cur_h).to(tl.int64)

#     off_chz = (bhid * N_CTX).to(tl.int64)

#     Q += adj_q
#     K += adj_k
#     V += adj_k
#     DO += adj_do
#     DQ += adj_q
#     DK += adj_dk
#     DV += adj_dk
#     M += off_chz
#     D += off_chz

#     offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_k = tl.arange(0, HEAD_DIM)
#     q_ptrs = Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
#     q = tl.load(q_ptrs)

#     dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
#     do = tl.load(DO + offs_m[:, None] * stride_dom +
#                  offs_k[None, :] * stride_dod)

#     # D (= delta) is pre-divided by ds_scale.
#     Di = tl.load(D + offs_m)

#     m = tl.load(M + offs_m)
#     m = m[:, None]

#     NUM_N_BLOCKS: tl.constexpr = BLOCK_N // BLOCK_M

#     for k_id in tl.static_range(TOP_K):
#         indices_ptrs = INDICES + cur_b * stride_ib + \
#             cur_h * stride_ih + pid * TOP_K + k_id
#         index = tl.load(indices_ptrs)
#         offs_n = index * BLOCK_N + tl.arange(0, BLOCK_N)

#     # for k_id in tl.static_range(0, TOP_K // NUM_N_BLOCKS):
#     #     offsets_in = k_id * NUM_N_BLOCKS + tl.arange(0, NUM_N_BLOCKS)
#     #     indices_ptrs = (INDICES + cur_b * stride_ib + cur_h * stride_ih +
#     #                     pid * stride_im + offsets_in * stride_ik)
#     #     index = tl.load(indices_ptrs)

#     #     offs_n = index[:, None] * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
#     #     offs_n = offs_n.reshape(BLOCK_N)

#         kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
#         vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd

#         kT = tl.load(kT_ptrs)
#         vT = tl.load(vT_ptrs)

#         qk = tl.dot(q, kT).to(tl.float32)
#         p = tl.math.exp(qk - m)
#         # Compute dP and dS.
#         dp = tl.dot(do, vT)
#         ds = p * (dp - Di[:, None])
#         # Compute dQ.
#         # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
#         dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

#         dvT = tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
#         dkT = tl.dot(tl.trans(ds).to(q.dtype), q) * scale
#         # dvT = tl.dot(tl.trans(p), do)
#         # dkT = tl.dot(tl.trans(ds), q.to(tl.float32)) * scale

#         kv_offs = offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd

#         tl.atomic_add(DK + kv_offs, dkT.to(tl.float32))
#         tl.atomic_add(DV + kv_offs, dvT.to(tl.float32))

#     dq_ptrs = DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
#     tl.store(dq_ptrs, dq)


@triton.jit
def _attn_bwd_dqkv(DQ, DK, DV,
                   Q, K, V,  #
                   DO, M, D, INDICES,
                   OFFSETS, FLAT_INDICES,
                   stride_qb, stride_qh, stride_qm, stride_qd,  #
                   stride_kb, stride_kh, stride_kn, stride_kd,  #
                   stride_dob, stride_doh, stride_dom, stride_dod,  #
                   stride_ib, stride_ih, stride_im, stride_ik,
                   stride_ob, stride_oh, stride_on,
                   stride_fib, stride_fih, stride_fink,
                   H, N_CTX, scale,
                   TOP_K: tl.constexpr,
                   BLOCK_M: tl.constexpr,  #
                   BLOCK_N: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr):
    bhid = tl.program_id(1)
    cur_b = bhid // H
    cur_h = bhid % H

    adj_q = (stride_qb * cur_b + stride_qh * cur_h).to(tl.int64)
    adj_do = (stride_dob * cur_b + stride_doh * cur_h).to(tl.int64)
    adj_k = (stride_kb * cur_b + stride_kh * cur_h).to(tl.int64)
    pid = tl.program_id(0)
    off_chz = (bhid * N_CTX).to(tl.int64)

    Q += adj_q
    K += adj_k
    V += adj_k
    DO += adj_do
    DQ += adj_q
    DK += adj_k
    DV += adj_k
    M += off_chz
    D += off_chz

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

    NUM_N_BLOCKS: tl.constexpr = BLOCK_N // BLOCK_M

    for k_id in tl.static_range(TOP_K):
        indices_ptrs = INDICES + cur_b * stride_ib + \
            cur_h * stride_ih + pid * TOP_K + k_id
        index = tl.load(indices_ptrs)
        offs_n = index * BLOCK_N + tl.arange(0, BLOCK_N)

        kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
        vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd

        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)

        qk = tl.dot(q, kT).to(tl.float32)
        p = tl.math.exp(qk - m)
        # Compute dP and dS.
        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds.to(kT.dtype), tl.trans(kT))

    dq_ptrs = DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qd
    tl.store(dq_ptrs, dq)

    start_offset = tl.load(OFFSETS + cur_b * stride_ob +
                           cur_h * stride_oh + pid * stride_on)
    end_offset = tl.load(OFFSETS + cur_b * stride_ob +
                         cur_h * stride_oh + (pid + 1) * stride_on)

    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd
    vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kd

    kT = tl.load(kT_ptrs)
    vT = tl.load(vT_ptrs)

    dkT = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    dvT = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

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
        # Compute dP and dS.
        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])

        dvT += tl.dot(tl.trans(p).to(q.dtype), do.to(q.dtype))
        dkT += tl.dot(tl.trans(ds).to(q.dtype), q) * scale

    kv_offs = offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd

    tl.store(DK + kv_offs, dkT)
    tl.store(DV + kv_offs, dvT)

# @profile
# def sparse_indices_attn_bwd(do, q, k, v, o, m, indices, topk, prev_size):
#     dq = torch.empty_like(q)
#     dk = torch.empty_like(k)
#     dv = torch.empty_like(v)
#     scale = 1 / math.sqrt(q.size(-1))

#     HEAD_DIM = q.shape[-1]

#     BATCH, N_HEAD, N_CTX = q.shape[:3]
#     PRE_BLOCK = min(64, N_CTX)
#     NUM_WARPS, NUM_STAGES = 4, 5
#     BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 64, 64, 64, 64
#     arg_k = k
#     arg_k = arg_k * scale
#     assert N_CTX % PRE_BLOCK == 0

#     assert dk.stride() == dv.stride()
#     assert k.stride() == v.stride()
#     pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
#     delta = torch.empty_like(m)
#     if not do.is_contiguous():
#         do = do.contiguous()
#     # print(o.shape, do.shape, delta.shape)
#     # print(o.stride(), do.stride(), delta.stride())

#     _attn_bwd_preprocess[pre_grid](
#         o, do, delta,
#         o.stride(0), o.stride(1), o.stride(2), o.stride(3),
#         do.stride(0), do.stride(1), do.stride(2), do.stride(3),
#         delta.stride(0), delta.stride(1), delta.stride(2), NUM_HEAD=N_HEAD,
#         BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM  #
#     )
#     # print(torch.abs((o*do).sum(-1) - delta).cpu().max())

#     grid = (N_CTX // 16, BATCH * N_HEAD)
#     _attn_bwd_dq[grid](
#         dq, q, arg_k, v, do, m, delta, indices,
#         q.stride(1), q.stride(2), q.stride(3),  #
#         k.stride(1), k.stride(2), k.stride(3),  #
#         N_HEAD, N_CTX, topk,
#         16, 16,
#         HEAD_DIM=HEAD_DIM,  #
#         num_warps=NUM_WARPS,  #
#         num_stages=NUM_STAGES  #
#     )

#     col_indices = indices.flatten(-2, -1).reshape(BATCH * N_HEAD, -1)
#     row_indices = torch.arange(prev_size)[:, None].expand(-1, topk).flatten(
#     )[None, :].expand(BATCH * N_HEAD, -1).to(col_indices.device)

#     crow_indices_tot = torch.empty(
#         BATCH*N_HEAD, prev_size + 1, dtype=torch.long, device=col_indices.device)
#     out_col_indices_tot = torch.empty(
#         BATCH*N_HEAD, prev_size * topk, dtype=torch.long, device=col_indices.device)

#     for bi in range(BATCH * N_HEAD):
#         coo_indices = torch.stack([row_indices[bi], col_indices[bi]])  # [2, N]
#         values = torch.ones(coo_indices.size(1))  # 占位值
#         sparse_tensor = torch.sparse_coo_tensor(
#             coo_indices, values, size=(prev_size, prev_size), device=col_indices.device)
#         sparse_tensor_T = sparse_tensor.transpose(0, 1)
#         csr_tensor = sparse_tensor_T.to_sparse_csr()
#         crow_indices = csr_tensor.crow_indices()
#         out_col_indices = csr_tensor.col_indices()
#         crow_indices_tot[bi] = crow_indices
#         out_col_indices_tot[bi] = out_col_indices

#     grid = (prev_size, BATCH * N_HEAD)
#     _attn_bwd_dkdv[grid](
#         dk, dv, q, arg_k, v, do, m, delta, crow_indices_tot, out_col_indices_tot,
#         q.stride(1), q.stride(2), q.stride(3),
#         k.stride(1), k.stride(2), k.stride(3),
#         crow_indices_tot.stride(0), crow_indices_tot.stride(1),
#         out_col_indices_tot.stride(0), out_col_indices_tot.stride(1),
#         16, 16, HEAD_DIM, N_CTX, scale
#     )

#     # print(do.max())
#     # print(dq.max())
#     # print(dk.max())
#     # print(dv.max())

#     return dq, dk, dv


# @profile
def sparse_indices_attn_scatter_bwd(do, q, k, v, o, m, indices, topk, block_size):
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k).to(torch.float32)
    dv = torch.zeros_like(v).to(torch.float32)
    scale = 1 / math.sqrt(q.size(-1))

    # NUM_GROUPS = 8
    # dk = torch.zeros(NUM_GROUPS, *k.shape,
    #                  device=k.device, dtype=torch.float32)
    # dv = torch.zeros_like(dk)

    # dk_32 = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    # dv_32 = torch.zeros(v.shape, dtype=torch.float32, device=v.device)

    HEAD_DIM = q.shape[-1]

    BATCH, N_HEAD, M_CTX = q.shape[:3]
    PRE_BLOCK = min(64, M_CTX)
    arg_k = k
    arg_k = arg_k * scale
    assert M_CTX % PRE_BLOCK == 0

    pre_grid = (M_CTX // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(m)
    if not do.is_contiguous():
        do = do.contiguous()
    # print(o.shape, do.shape, delta.shape)
    # print(o.stride(), do.stride(), delta.stride())

    # print(o.stride())
    # print(do.stride())
    # print(delta.stride())
    # print(q.stride(), k.stride(), v.stride())
    # print(q.shape, k.shape, v.shape)
    # print(dq.stride(), dk_32.stride(), dv_32.stride())
    # print(m.stride())
    # print(indices.stride())
    # exit(0)

    _attn_bwd_preprocess[pre_grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2), NUM_HEAD=N_HEAD,
        BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM  #
    )
    # print(torch.abs((o*do).sum(-1) - delta).cpu().max())

    # debug = torch.zeros((16, 16), dtype=torch.float32, device=q.device)
    # debug2 = torch.zeros((16, 64), dtype=torch.float32, device=q.device)
    # print(k.shape, HEAD_DIM)

    grid = (M_CTX // block_size, BATCH * N_HEAD)

    _attn_bwd_dqkv_scatter[grid](
        # _attn_bwd_dqkv_scatter_group[grid](
        dq, dk, dv, q, arg_k, v, do, m, delta, indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),  #
        indices.stride(0), indices.stride(
            1), indices.stride(2), indices.stride(3),
        N_HEAD, M_CTX, scale,
        topk,
        block_size, block_size,
        HEAD_DIM=HEAD_DIM
        # num_warps=4,  #
        # num_stages=2
    )

    # dk = dk.sum(0)
    # dv = dv.sum(0)
    # dk = dk[0]
    # dv = dv[0]

    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)

    return dq, dk, dv


# @profile
def sparse_indices_attn_scatter_bwd_v2(do, q, k, v, o, m, indices,
                                       topk, block_size):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    scale = 1 / math.sqrt(q.size(-1))

    offsets, flat_indices = transpose_indices(indices, k.shape[-2])

    HEAD_DIM = q.shape[-1]

    BATCH, N_HEAD, M_CTX = q.shape[:3]
    PRE_BLOCK = min(64, M_CTX)
    NUM_WARPS, NUM_STAGES = 4, 5
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 64, 64, 64, 64
    arg_k = k
    arg_k = arg_k * scale
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
        BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM  #
    )

    grid = (M_CTX // block_size, BATCH * N_HEAD)

    _attn_bwd_dqkv[grid](
        dq, dk, dv, q, arg_k, v, do, m, delta, indices,
        offsets, flat_indices,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),  #
        indices.stride(0), indices.stride(
            1), indices.stride(2), indices.stride(3),
        offsets.stride(0), offsets.stride(1), offsets.stride(2),
        flat_indices.stride(0), flat_indices.stride(1), flat_indices.stride(2),
        N_HEAD, M_CTX, scale,
        topk,
        block_size, block_size,
        HEAD_DIM=HEAD_DIM
        # num_warps=4,  #
        # num_stages=2
    )

    # dk = dk.sum(0)
    # dv = dv.sum(0)
    # dk = dk[0]
    # dv = dv[0]

    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)

    return dq, dk, dv
