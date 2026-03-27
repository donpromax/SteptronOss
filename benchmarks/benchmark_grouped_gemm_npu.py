from steptronoss.utils.npu_patch import apply_npu_patch

apply_npu_patch()

import argparse
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT.parent))


DEFAULT_PARAM_SETS = [
    {
        "name": "moe_like_large",
        "group_size": 36,
        "batch_size": 3256,
        "k": 4096,
        "n": 2560,
        "dtype": "bf16",
        "warmup": 20,
        "iters": 20,
        "trans_b": True,
    },
]


def _dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def _sync():
    torch.npu.synchronize()


def _build_inputs(params: dict[str, object], device: torch.device):
    dtype = _dtype_from_name(params["dtype"])
    group_size = int(params["group_size"])
    batch_size = int(params["batch_size"])
    k = int(params["k"])
    n = int(params["n"])
    trans_b = bool(params["trans_b"])

    batch_sizes = torch.full((group_size,), batch_size, device=device, dtype=torch.int64)
    total_m = int(batch_sizes.sum().item())

    a = torch.randn(total_m, k, device=device, dtype=dtype, requires_grad=True)
    if trans_b:
        b = torch.randn(group_size, n, k, device=device, dtype=dtype, requires_grad=True)
    else:
        b = torch.randn(group_size, k, n, device=device, dtype=dtype, requires_grad=True)
    return a, b, batch_sizes


def _run_baseline(
    mat_a_flat: torch.Tensor, mat_b: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool
) -> torch.Tensor:
    batch_sizes_list = batch_sizes.tolist()
    outputs = []
    start = 0
    for i, size in enumerate(batch_sizes_list):
        rhs = mat_b[i].t() if trans_b else mat_b[i]
        outputs.append(mat_a_flat[start : start + size] @ rhs)
        start += size
    if outputs:
        return torch.cat(outputs, dim=0)
    return mat_a_flat.new_zeros((0, mat_b.shape[1] if trans_b else mat_b.shape[2]))


def _run_npu_gmm_v2(
    mat_a_flat: torch.Tensor, mat_b: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool
) -> torch.Tensor:
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


def _time_forward(fn, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(warmup):
        with torch.no_grad():
            out = fn()
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            out = fn()
    _sync()
    return (time.perf_counter() - start) * 1000.0 / iters, out


def _time_forward_backward(
    fn, a: torch.Tensor, b: torch.Tensor, warmup: int, iters: int
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn()
        out.sum().backward()
        a.grad = None
        b.grad = None
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        out.sum().backward()
        grad_a = a.grad.detach().clone()
        grad_b = b.grad.detach().clone()
        a.grad = None
        b.grad = None
    _sync()
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    return total_ms, out.detach().clone(), grad_a, grad_b


def _max_abs_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x.float() - y.float()).abs().max().item())


def _check_close(x: torch.Tensor, y: torch.Tensor, rtol: float, atol: float) -> bool:
    try:
        torch.testing.assert_close(x, y, rtol=rtol, atol=atol)
        return True
    except Exception:
        return False


def _bench_one(params: dict[str, object], rtol: float, atol: float):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is not available.")

    device = torch.device("npu")
    warmup = int(params["warmup"])
    iters = int(params["iters"])
    trans_b = bool(params["trans_b"])

    a_base, b_base, batch_sizes = _build_inputs(params, device)
    a_npu = a_base.detach().clone().requires_grad_(True)
    b_npu = b_base.detach().clone().requires_grad_(True)

    fw_ms_base, ref_out = _time_forward(
        lambda: _run_baseline(a_base, b_base, batch_sizes, trans_b),
        warmup=warmup,
        iters=iters,
    )
    total_ms_base, ref_out_bw, ref_da, ref_db = _time_forward_backward(
        lambda: _run_baseline(a_base, b_base, batch_sizes, trans_b),
        a=a_base,
        b=b_base,
        warmup=warmup,
        iters=iters,
    )

    fw_ms_npu, out_npu = _time_forward(
        lambda: _run_npu_gmm_v2(a_npu, b_npu, batch_sizes, trans_b),
        warmup=warmup,
        iters=iters,
    )
    total_ms_npu, out_npu_bw, da_npu, db_npu = _time_forward_backward(
        lambda: _run_npu_gmm_v2(a_npu, b_npu, batch_sizes, trans_b),
        a=a_npu,
        b=b_npu,
        warmup=warmup,
        iters=iters,
    )

    print(
        f"[npu_grouped_gemm] name={params['name']} group={params['group_size']} "
        f"batch={params['batch_size']} k={params['k']} n={params['n']} "
        f"dtype={params['dtype']} trans_b={trans_b}"
    )
    print("backend, fw_ms, bw_ms, total_ms")
    print(f"baseline, {fw_ms_base:.3f}, {total_ms_base - fw_ms_base:.3f}, {total_ms_base:.3f}")
    print(f"npu_gmm_v2, {fw_ms_npu:.3f}, {total_ms_npu - fw_ms_npu:.3f}, {total_ms_npu:.3f}")
    print(
        "speedup_vs_baseline, "
        f"fw={fw_ms_base / fw_ms_npu:.2f}x, "
        f"bw={(total_ms_base - fw_ms_base) / (total_ms_npu - fw_ms_npu):.2f}x, "
        f"total={total_ms_base / total_ms_npu:.2f}x"
    )
    print("metric, close, max_abs_diff")
    print(f"forward, {_check_close(out_npu, ref_out, rtol, atol)}, {_max_abs_diff(out_npu, ref_out):.6f}")
    print(
        f"forward_bw_run, {_check_close(out_npu_bw, ref_out_bw, rtol, atol)}, {_max_abs_diff(out_npu_bw, ref_out_bw):.6f}"
    )
    print(f"grad_a, {_check_close(da_npu, ref_da, rtol, atol)}, {_max_abs_diff(da_npu, ref_da):.6f}")
    print(f"grad_b, {_check_close(db_npu, ref_db, rtol, atol)}, {_max_abs_diff(db_npu, ref_db):.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    torch.npu.set_device(0)
    for params in DEFAULT_PARAM_SETS:
        _bench_one(params, rtol=args.rtol, atol=args.atol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
