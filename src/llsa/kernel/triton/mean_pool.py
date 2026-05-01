import torch
import triton
import triton.language as tl


@triton.jit
def _mean_pool1d_bhsc_fwd(X, Y,
                          stride_xb, stride_xh, stride_xs, stride_xc,
                          stride_yb, stride_yh, stride_ys, stride_yc,
                          s_ctx,
                          K,
                          NUM_HEAD: tl.constexpr,
                          HEAD_DIM: tl.constexpr):
    pid_s = tl.program_id(axis=0)

    pid_bh = tl.program_id(axis=1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    offs_c = tl.arange(0, HEAD_DIM)

    x_ptrs = X + cur_b * stride_xb + cur_h * stride_xh + \
        (pid_s*K) * stride_xs + offs_c * stride_xc
    y_ptrs = Y + cur_b * stride_yb + cur_h * \
        stride_yh + pid_s * stride_ys + offs_c * stride_yc

    acc = tl.zeros([HEAD_DIM], X.dtype.element_ty)
    seg_start = pid_s * K
    valid_len = tl.minimum(K, s_ctx - seg_start)
    for t in range(K):
        mask_s = (seg_start + t) < s_ctx
        acc += tl.load(x_ptrs, mask=mask_s, other=0.0)
        x_ptrs += stride_xs
    acc = acc / valid_len
    tl.store(y_ptrs, acc)


@triton.jit
def _mean_pool1d_bhsc_bwd(dY, dX,
                          stride_dyb, stride_dyh, stride_dys, stride_dyc,
                          stride_dxb, stride_dxh, stride_dxs, stride_dxc,
                          s_ctx,
                          K,
                          NUM_HEAD: tl.constexpr,
                          HEAD_DIM: tl.constexpr):
    pid_s = tl.program_id(axis=0)

    pid_bh = tl.program_id(axis=1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    offs_c = tl.arange(0, HEAD_DIM)

    dx_ptrs = dX + cur_b * stride_dxb + cur_h * stride_dxh + \
        (pid_s*K) * stride_dxs + offs_c * stride_dxc
    dy_ptrs = dY + cur_b * stride_dyb + cur_h * \
        stride_dyh + pid_s * stride_dys + offs_c * stride_dyc

    seg_start = pid_s * K
    valid_len = tl.minimum(K, s_ctx - seg_start)
    grad = tl.load(dy_ptrs) / valid_len

    for t in range(K):
        mask_s = (seg_start + t) < s_ctx
        tl.store(dx_ptrs, grad, mask=mask_s)
        dx_ptrs += stride_dxs


@triton.jit
def _mean_pool1d_bsc_fwd(X, Y,
                         stride_xb, stride_xs, stride_xc,
                         stride_yb, stride_ys, stride_yc,
                         s_ctx,
                         c_ctx,
                         K,
                         HEAD_BLOCK: tl.constexpr):
    pid_c = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    offs_c = pid_c * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    mask_c = offs_c < c_ctx

    x_ptrs = X + pid_b * stride_xb + \
        (pid_s*K) * stride_xs + offs_c * stride_xc
    y_ptrs = Y + pid_b * stride_yb + \
        pid_s * stride_ys + offs_c * stride_yc

    acc = tl.zeros([HEAD_BLOCK], X.dtype.element_ty)
    seg_start = pid_s * K
    valid_len = tl.minimum(K, s_ctx - seg_start)
    for t in range(K):
        mask_s = (seg_start + t) < s_ctx
        acc += tl.load(x_ptrs, mask=mask_c & mask_s, other=0.0)
        x_ptrs += stride_xs
    acc = acc / valid_len
    tl.store(y_ptrs, acc, mask=mask_c)


@triton.jit
def _mean_pool1d_bsc_bwd(dY, dX,
                         stride_dyb, stride_dys, stride_dyc,
                         stride_dxb, stride_dxs, stride_dxc,
                         s_ctx,
                         c_ctx,
                         K,
                         HEAD_BLOCK: tl.constexpr):
    pid_c = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    offs_c = pid_c * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    mask_c = offs_c < c_ctx

    dx_ptrs = dX + pid_b * stride_dxb + \
        (pid_s*K) * stride_dxs + offs_c * stride_dxc
    dy_ptrs = dY + pid_b * stride_dyb + \
        pid_s * stride_dys + offs_c * stride_dyc

    seg_start = pid_s * K
    valid_len = tl.minimum(K, s_ctx - seg_start)
    grad = tl.load(dy_ptrs, mask=mask_c, other=0) / valid_len

    for t in range(K):
        mask_s = (seg_start + t) < s_ctx
        tl.store(dx_ptrs, grad, mask=mask_c & mask_s)
        dx_ptrs += stride_dxs


def mean_pool1d_bhsc_fwd(x, k):
    B, H, S, C = x.shape
    S_out = triton.cdiv(S, k)
    y = torch.empty((B, H, S_out, C), device=x.device, dtype=x.dtype)

    grid = (S_out, B * H)
    _mean_pool1d_bhsc_fwd[grid](
        x, y,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        S, k, H, C
    )
    return y


class MeanPool1dFunction_BHSC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size: int):
        B, H, S, C = x.shape
        S_out = triton.cdiv(S, kernel_size)
        y = torch.empty((B, H, S_out, C),
                        device=x.device, dtype=x.dtype)

        # meta‑params
        grid = (S_out, B * H)

        _mean_pool1d_bhsc_fwd[grid](
            x, y,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            S, kernel_size, H, C
        )

        ctx.kernel_size = kernel_size
        ctx.seq_len = S
        return y

    @staticmethod
    def backward(ctx, dy):
        K = ctx.kernel_size
        S = ctx.seq_len
        B, H, S_out, C = dy.shape
        dx = torch.empty((B, H, S, C),
                         device=dy.device, dtype=dy.dtype)

        grid = (S_out, B * H)

        _mean_pool1d_bhsc_bwd[grid](
            dy, dx,
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            S, K, H, C
        )
        return dx, None


class MeanPool1dFunction_BSC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size: int):
        B, S, C = x.shape
        S_out = triton.cdiv(S, kernel_size)
        y = torch.empty((B, S_out, C),
                        device=x.device, dtype=x.dtype)

        # meta‑params
        HEAD_BLOCK = 64
        grid = (triton.cdiv(C, HEAD_BLOCK), S_out, B)

        _mean_pool1d_bsc_fwd[grid](
            x, y,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            S, C, kernel_size, HEAD_BLOCK
        )

        ctx.kernel_size = kernel_size
        ctx.seq_len = S
        return y

    @staticmethod
    def backward(ctx, dy):
        K = ctx.kernel_size
        S = ctx.seq_len
        B, S_out, C = dy.shape
        dx = torch.empty((B, S, C),
                         device=dy.device, dtype=dy.dtype)

        HEAD_BLOCK = 64
        grid = (triton.cdiv(C, HEAD_BLOCK), S_out, B)

        _mean_pool1d_bsc_bwd[grid](
            dy, dx,
            dy.stride(0), dy.stride(1), dy.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            S, C, K, HEAD_BLOCK
        )
        return dx, None


def mean_pool1d(x: torch.Tensor, k: int, format='bhsc'):
    if format == 'bhsc':
        return MeanPool1dFunction_BHSC.apply(x, k)
    elif format == 'bsc':
        return MeanPool1dFunction_BSC.apply(x, k)
    else:
        raise NotImplementedError(
            f'Formar {format} is not implemented for my mean_pool1d')
