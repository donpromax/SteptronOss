from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_weighted_gather_kernel(
    in_ptr,
    index_ptr,
    weight_ptr,
    out_ptr,
    token_num,
    top_k,
    hidden_dim,
    stride_in0,
    stride_in1,
    stride_idx0,
    stride_idx1,
    stride_w0,
    stride_w1,
    stride_out0,
    stride_out1,
    ACC_FP32: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= token_num:
        return

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_dim

    acc_dtype = tl.float32 if ACC_FP32 else IN_DTYPE
    acc = tl.zeros((BLOCK_H,), dtype=acc_dtype)

    for k in range(0, top_k):
        idx = tl.load(index_ptr + pid_t * stride_idx0 + k * stride_idx1)
        idx = idx.to(tl.int32)
        valid = idx >= 0
        idx = tl.where(valid, idx, 0)

        w = tl.load(weight_ptr + pid_t * stride_w0 + k * stride_w1)
        w = w.to(acc_dtype)
        w = tl.where(valid, w, 0)

        in_ptrs = in_ptr + idx * stride_in0 + offs_h * stride_in1
        vals = tl.load(in_ptrs, mask=mask_h, other=0.0).to(acc_dtype)
        acc += vals * w

    out = acc.to(IN_DTYPE)
    out_ptrs = out_ptr + pid_t * stride_out0 + offs_h * stride_out1
    tl.store(out_ptrs, out, mask=mask_h)


class _TritonMoEWeightedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError("index must be int32 or int64 tensor")
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("weight must be a floating tensor")
        if index.dim() != 2 or weight.dim() != 2:
            raise ValueError("index and weight must be 2D [token_num, top_k]")

        token_num, top_k = index.shape
        if weight.shape != index.shape:
            raise ValueError("index and weight must have the same shape")
        hidden_dim = input.shape[-1]

        out = input.new_zeros((token_num, hidden_dim))
        if token_num == 0 or index.numel() == 0:
            ctx.save_for_backward(input, index, weight)
            return out

        acc_fp32 = weight.dtype == torch.float32
        in_dtype = {
            torch.float16: tl.float16,
            torch.bfloat16: tl.bfloat16,
            torch.float32: tl.float32,
        }.get(input.dtype)
        if in_dtype is None:
            raise TypeError("input must be fp16/bf16/fp32")

        BLOCK_H = 128
        grid = (token_num, triton.cdiv(hidden_dim, BLOCK_H))
        _moe_weighted_gather_kernel[grid](
            input,
            index,
            weight,
            out,
            token_num,
            top_k,
            hidden_dim,
            input.stride(0),
            input.stride(1),
            index.stride(0),
            index.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            ACC_FP32=acc_fp32,
            IN_DTYPE=in_dtype,
            BLOCK_H=BLOCK_H,
        )

        ctx.save_for_backward(input, index, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        input, index, weight = ctx.saved_tensors
        token_num, top_k = index.shape
        hidden_dim = input.shape[-1]

        flat_index = index.reshape(-1)
        valid = flat_index >= 0
        safe_index = flat_index.clamp_min(0).to(torch.int64)
        gathered = input.index_select(0, safe_index).reshape(token_num, top_k, hidden_dim)
        gathered = gathered * valid.reshape(token_num, top_k, 1).to(gathered.dtype)

        grad_weight = (grad_out.unsqueeze(1) * gathered).sum(dim=2)
        if grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(weight.dtype)

        grad_in = input.new_zeros(input.shape)
        grad_contrib = (grad_out.unsqueeze(1) * weight.to(grad_out.dtype).unsqueeze(-1)).reshape(-1, hidden_dim)
        grad_contrib = grad_contrib * valid.to(grad_contrib.dtype).unsqueeze(-1)
        grad_in.index_add_(0, safe_index, grad_contrib.to(grad_in.dtype))

        return grad_in, None, grad_weight


def triton_moe_weighted_gather(input: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _TritonMoEWeightedGather.apply(input, index, weight)
