import torch
import triton
import triton.language as tl


@triton.jit
def _count_k_major(INDICES, COUNTS,
                   stride_ib, stride_ih, stride_im, stride_ik,
                   stride_cb, stride_ch, stride_cn,
                   TOPK: tl.constexpr,
                   NUM_HEAD: tl.constexpr,
                   ):

    bhid = tl.program_id(1)
    cur_b = bhid // NUM_HEAD
    cur_h = bhid % NUM_HEAD
    pid_m = tl.program_id(0)

    indices_ptr = INDICES + cur_b * stride_ib + \
        cur_h * stride_ih + pid_m * stride_im
    counts_ptr = COUNTS + cur_b * stride_cb + cur_h * stride_ch

    for k_id in tl.static_range(TOPK):
        n_block_idx = tl.load(indices_ptr + k_id * stride_ik)

        c_count_ptr = counts_ptr + (n_block_idx + 1) * stride_cn
        tl.atomic_add(c_count_ptr, 1)


def count_k_major(indices, k_ntx):
    B, H, M, TOPK = indices.shape

    count = torch.zeros(B, H, 1 + k_ntx, dtype=torch.uint32,
                        device=indices.device)

    grid = (M, B * H)
    _count_k_major[grid](
        indices,
        count,
        *indices.stride(),
        *count.stride(),
        TOPK, H
    )

    return count


@triton.jit
def _k_major_indices(INDICES, OFFSETS,
                     FLAT_INDICES, COUNTS,
                     stride_ib, stride_ih, stride_im, stride_ik,
                     stride_ob, stride_oh, stride_on,
                     stride_fb, stride_fh, stride_fnk,
                     stride_cb, stride_ch, stride_cn,
                     TOPK: tl.constexpr,
                     NUM_HEAD: tl.constexpr,
                     ):

    bhid = tl.program_id(1)
    cur_b = bhid // NUM_HEAD
    cur_h = bhid % NUM_HEAD
    pid_m = tl.program_id(0)

    indices_ptr = INDICES + cur_b * stride_ib + \
        cur_h * stride_ih + pid_m * stride_im
    offsets_ptr = OFFSETS + cur_b * stride_ob + cur_h * stride_oh
    f_indices_ptr = FLAT_INDICES + cur_b * stride_fb + cur_h * stride_fh
    counts_ptr = COUNTS + cur_b * stride_cb + cur_h * stride_ch

    for k_id in tl.static_range(TOPK):
        n_block_idx = tl.load(indices_ptr + k_id * stride_ik)

        c_count_ptr = counts_ptr + n_block_idx * stride_cn
        i = tl.atomic_add(c_count_ptr, 1)

        start_offset = tl.load(offsets_ptr + n_block_idx * stride_on)
        write_ptr = f_indices_ptr + (start_offset + i) * stride_fnk
        tl.store(write_ptr, pid_m)


def k_major_indices(indices, offsets, k_ntx):
    B, H, M, TOPK = indices.shape

    count = torch.zeros(B, H, k_ntx, dtype=torch.uint32, device=indices.device)

    flat_indices = torch.empty(
        B, H, M * TOPK, dtype=torch.uint32, device=indices.device)

    grid = (M, B * H)

    _k_major_indices[grid](
        indices, offsets,
        flat_indices, count,
        *indices.stride(),
        *offsets.stride(),
        *flat_indices.stride(),
        *count.stride(),
        TOPK, H
    )

    return flat_indices


def transpose_indices(q_topk_indices, k_ntx=None):
    if k_ntx is None:
        k_ntx = q_topk_indices.shape[-2]
    count = count_k_major(q_topk_indices, k_ntx)
    offsets = torch.cumsum(count, -1)
    return offsets, k_major_indices(q_topk_indices, offsets, k_ntx)
