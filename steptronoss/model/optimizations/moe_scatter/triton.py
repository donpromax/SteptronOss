from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_scatter_kernel(
    in_ptr,
    index_ptr,
    out_ptr,
    n_elements,
    hidden_dim,
    out_size,
    stride_in0,
    stride_in1,
    stride_out0,
    stride_out1,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    n = pid_n
    if n >= n_elements:
        return

    idx = tl.load(index_ptr + n).to(tl.int32)
    if idx < 0 or idx >= out_size:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_dim

    in_ptrs = in_ptr + n * stride_in0 + offs_h * stride_in1
    out_ptrs = out_ptr + idx * stride_out0 + offs_h * stride_out1
    vals = tl.load(in_ptrs, mask=mask_h, other=0.0)
    tl.store(out_ptrs, vals, mask=mask_h)


class _TritonMoEScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, index: torch.Tensor, out_size: int | None = None) -> torch.Tensor:
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError("index must be int32 or int64 tensor")
        if index.dim() != 2:
            raise ValueError("index must be 2D [token_num, top_k]")
        if input.dim() != 2:
            raise ValueError("input must be 2D [token_num, hidden_dim]")

        token_num, top_k = index.shape
        if input.shape[0] != token_num:
            raise ValueError("input and index must have the same token_num")

        hidden_dim = input.shape[-1]
        if out_size is None:
            out_size = int((index >= 0).sum().item())
        else:
            out_size = int(out_size)
        out = input.new_zeros((out_size, hidden_dim))

        if out_size == 0 or index.numel() == 0:
            ctx.save_for_backward(index)
            ctx.hidden_dim = hidden_dim
            return out

        flat_index = index.reshape(-1).contiguous()
        flat_input = input.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim).contiguous()

        n_elements = flat_index.numel()
        BLOCK_H = 128
        grid = (n_elements, triton.cdiv(hidden_dim, BLOCK_H))

        _moe_scatter_kernel[grid](
            flat_input,
            flat_index,
            out,
            n_elements,
            hidden_dim,
            out_size,
            flat_input.stride(0),
            flat_input.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_H=BLOCK_H,
        )

        ctx.save_for_backward(index)
        ctx.hidden_dim = hidden_dim
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (index,) = ctx.saved_tensors
        token_num, top_k = index.shape
        hidden_dim = ctx.hidden_dim

        if grad_out.numel() == 0:
            grad_input = grad_out.new_zeros((token_num, hidden_dim))
            return grad_input, None, None

        flat_index = index.reshape(-1)
        valid = flat_index >= 0
        if valid.all():
            gathered = grad_out.index_select(0, flat_index.to(torch.int64))
            grad_input = gathered.reshape(token_num, top_k, hidden_dim).sum(dim=1)
        else:
            grad_input = grad_out.new_zeros((token_num, hidden_dim))
            if valid.any():
                gathered = grad_out.index_select(0, flat_index[valid].to(torch.int64))
                tmp = grad_out.new_zeros((flat_index.numel(), hidden_dim))
                tmp[valid] = gathered
                grad_input = tmp.reshape(token_num, top_k, hidden_dim).sum(dim=1)

        return grad_input, None, None


def triton_moe_scatter(input: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return _TritonMoEScatter.apply(input, index, None)


def triton_moe_scatter_with_size(input: torch.Tensor, index: torch.Tensor, out_size: int) -> torch.Tensor:
    return _TritonMoEScatter.apply(input, index, out_size)
