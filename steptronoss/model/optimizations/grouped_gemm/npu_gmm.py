import torch


def mindspeed_npu_grouped_gemm_v2(mat_a_flat, mat_b, batch_sizes, trans_b=False):
    try:
        from mindspeed.ops.gmm import npu_gmm_v2
    except Exception as exc:
        raise ImportError("from mindspeed.ops.gmm import npu_gmm_v2 failed.") from exc

    if mat_a_flat.shape[0] == 0:
        return mat_a_flat.new_zeros((0, mat_b.shape[1] if trans_b else mat_b.shape[2]))

    weight = mat_b.transpose(-1, -2) if trans_b else mat_b
    if batch_sizes.device.type != "npu":
        batch_sizes = batch_sizes.to(device=mat_a_flat.device)
    batch_sizes = batch_sizes.to(dtype=torch.int64)
    return npu_gmm_v2(mat_a_flat, weight, bias=None, group_list=batch_sizes, group_type=0)
