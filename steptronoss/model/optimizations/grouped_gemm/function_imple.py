from __future__ import annotations

import torch


class FunctionImpleGroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mat_a_flat, mat_b, batch_sizes, trans_b):
        batch_sizes_list = batch_sizes.tolist()
        outputs = []
        start = 0
        for i, size in enumerate(batch_sizes_list):
            rhs = mat_b[i].t() if trans_b else mat_b[i]
            outputs.append(mat_a_flat[start : start + size] @ rhs)
            start += size
        if outputs:
            outs = torch.cat(outputs, dim=0)
        else:
            outs = mat_a_flat.new_zeros((0, mat_b.shape[1] if trans_b else mat_b.shape[2]))
        ctx.save_for_backward(mat_a_flat, mat_b, batch_sizes)
        ctx.trans_b = trans_b
        return outs

    @staticmethod
    def backward(ctx, grad_out):
        mat_a_flat, mat_b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        if grad_out.numel() == 0:
            return (
                mat_a_flat.new_zeros(mat_a_flat.shape),
                mat_b.new_zeros(mat_b.shape),
                None,
                None,
            )

        device = mat_a_flat.device
        batch_sizes = batch_sizes.to(device=device, dtype=torch.int32)
        group_size = batch_sizes.numel()
        if group_size == 0:
            return (
                mat_a_flat.new_zeros(mat_a_flat.shape),
                mat_b.new_zeros(mat_b.shape),
                None,
                None,
            )

        starts = torch.cumsum(batch_sizes, dim=0) - batch_sizes
        starts = starts.to(torch.int64)

        grad_a = mat_a_flat.new_empty(mat_a_flat.shape)
        grad_b = mat_b.new_empty(mat_b.shape)

        for i in range(group_size):
            size = int(batch_sizes[i].item())
            if size == 0:
                grad_b[i].zero_()
                continue
            start = int(starts[i].item())
            a_slice = mat_a_flat[start : start + size]
            grad_slice = grad_out[start : start + size]
            if trans_b:
                b_slice = mat_b[i]
                grad_a[start : start + size] = grad_slice @ b_slice
                grad_b[i] = grad_slice.t() @ a_slice
            else:
                b_slice = mat_b[i]
                grad_a[start : start + size] = grad_slice @ b_slice.t()
                grad_b[i] = a_slice.t() @ grad_slice

        return grad_a, grad_b, None, None


def function_imple_grouped_gemm(mat_a_flat, mat_b, batch_sizes, trans_b=False):
    return FunctionImpleGroupedGemm.apply(mat_a_flat, mat_b, batch_sizes, trans_b)
