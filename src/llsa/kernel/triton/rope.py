import torch
import triton
import triton.language as tl


@triton.jit
def _my_rope_fwd(X, Y, FREQ,
                 stride_xb, stride_xh, stride_xs, stride_xc,
                 stride_yb, stride_yh, stride_ys, stride_yc,
                 stride_fs, stride_fc,
                 BLOCK_S: tl.constexpr,
                 NUM_HEAD: tl.constexpr,
                 HEAD_DIM: tl.constexpr):
    pid_s = tl.program_id(axis=0)

    pid_bh = tl.program_id(axis=1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    x_row = X + cur_b * stride_xb + cur_h * stride_xh + \
        offs_s * stride_xs
    y_row = Y + cur_b * stride_yb + cur_h * stride_yh + \
        offs_s * stride_ys
    freq_row = FREQ + offs_s * stride_fs

    idx_even = tl.arange(0, HEAD_DIM // 2) * 2
    idx_odd = idx_even + 1

    x_even_ptr = x_row[:, None] + idx_even[None, :] * stride_xc
    x_odd_ptr = x_row[:, None] + idx_odd[None, :] * stride_xc

    y_even_ptr = y_row[:, None] + idx_even[None, :] * stride_yc
    y_odd_ptr = y_row[:, None] + idx_odd[None, :] * stride_yc

    sin_ptr = freq_row[:, None] + idx_even[None, :] * stride_fc
    cos_ptr = freq_row[:, None] + idx_odd[None, :] * stride_fc

    x_even = tl.load(x_even_ptr)
    x_odd = tl.load(x_odd_ptr)

    sin_f = tl.load(sin_ptr)
    cos_f = tl.load(cos_ptr)

    y_even = cos_f * x_even - sin_f * x_odd
    y_odd = sin_f * x_even + cos_f * x_odd

    tl.store(y_even_ptr, y_even)
    tl.store(y_odd_ptr,  y_odd)


@triton.jit
def _my_rope_bwd(DO, FREQ, DX,
                 stride_dob, stride_doh, stride_dos, stride_doc,
                 stride_xb, stride_xh, stride_xs, stride_xc,
                 stride_fs, stride_fc,
                 BLOCK_S: tl.constexpr,
                 NUM_HEAD: tl.constexpr,
                 HEAD_DIM: tl.constexpr):
    pid_s = tl.program_id(axis=0)

    pid_bh = tl.program_id(axis=1)
    cur_b = pid_bh // NUM_HEAD
    cur_h = pid_bh % NUM_HEAD

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    do_row = DO + cur_b * stride_dob + cur_h * stride_doh + offs_s * stride_dos
    freq_row = FREQ + offs_s * stride_fs

    dx_row = DX + cur_b * stride_xb + cur_h * stride_xh + \
        offs_s * stride_xs

    idx_even = tl.arange(0, HEAD_DIM // 2) * 2
    idx_odd = idx_even + 1

    dx_even_ptr = dx_row[:, None] + idx_even[None, :] * stride_xc
    dx_odd_ptr = dx_row[:, None] + idx_odd[None, :] * stride_xc

    do_even_ptr = do_row[:, None] + idx_even[None, :] * stride_doc
    do_odd_ptr = do_row[:, None] + idx_odd[None, :] * stride_doc

    sin_ptr = freq_row[:, None] + idx_even[None, :] * stride_fc
    cos_ptr = freq_row[:, None] + idx_odd[None, :] * stride_fc

    do_even = tl.load(do_even_ptr)
    do_odd = tl.load(do_odd_ptr)

    sin_f = tl.load(sin_ptr)
    cos_f = tl.load(cos_ptr)

    dx_even = cos_f * do_even + sin_f * do_odd
    dx_odd = -sin_f * do_even + cos_f * do_odd

    tl.store(dx_even_ptr, dx_even)
    tl.store(dx_odd_ptr,  dx_odd)


class MyRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freq):

        assert x.dim() == 4 and freq.dim() == 2, "x:[B,H,S,D], freq:[S,D]"
        B, H, S, D = x.shape
        assert D % 2 == 0, "D must be even (pairs of cos/sin)."
        assert freq.shape[0] == S and freq.shape[1] == D

        y = torch.empty_like(x)

        BLOCK_S = 1
        grid = (S // BLOCK_S, B * H)

        _my_rope_fwd[grid](
            x, y, freq,
            *x.stride(),
            *y.stride(),
            *freq.stride(),
            BLOCK_S, H, D,
            num_warps=4
        )

        # save for backward
        ctx.save_for_backward(freq)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        freq, = ctx.saved_tensors
        dx = torch.empty_like(grad_output)

        B, H, S, D = dx.shape
        BLOCK_S = 1
        grid = (S // BLOCK_S, B * H)

        _my_rope_bwd[grid](
            grad_output, freq, dx,
            *grad_output.stride(),
            *dx.stride(),
            *freq.stride(),
            BLOCK_S, H, D,
            num_warps=4
        )

        return dx, None


my_rope_fn = MyRopeFunction.apply
