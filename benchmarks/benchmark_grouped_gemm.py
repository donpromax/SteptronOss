import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from benchmarks.common import run_with_backends
from steptronoss.model.optimizations.grouped_gemm import triton as gmm_mod
from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER

PARAM_SETS = [
    {
        "group_size": 36,
        "batch_size": 3256,
        "k": 4096,
        "n": 2560,
        "dtype": "bf16",
        "warmup": 20,
        "iters": 20,
        "check": True,
    },
]


def _baseline_grouped_gemm(mat_a_flat: torch.Tensor, mat_b: torch.Tensor, batch_sizes: torch.Tensor) -> torch.Tensor:
    batch_sizes_list = batch_sizes.tolist()
    outputs = []
    start = 0
    for i, size in enumerate(batch_sizes_list):
        rhs = mat_b[i]
        outputs.append(mat_a_flat[start : start + size] @ rhs)
        start += size
    if outputs:
        return torch.cat(outputs, dim=0)
    return mat_a_flat.new_zeros((0, mat_b.shape[1]))


def _triton_dtype(tensor: torch.Tensor):
    if tensor.dtype == torch.float16:
        return gmm_mod.tl.float16
    if tensor.dtype == torch.bfloat16:
        return gmm_mod.tl.bfloat16
    if tensor.dtype == torch.float32:
        return gmm_mod.tl.float32
    return None


def _bench_variant(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    function_imple: str | None,
    warmup: int,
    iters: int,
    check_correctness: bool,
    ref_out: torch.Tensor | None,
    ref_da: torch.Tensor | None,
    ref_db: torch.Tensor | None,
):
    def _forward_only():
        from steptronoss.model.utils.moe_utils import grouped_gemm

        return grouped_gemm(a, b, batch_sizes)
        # return _GroupedGemmBenchFn.apply(
        #     a, b, batch_sizes, use_fw_kernel, use_grad_a_kernel, use_grad_b_kernel
        # )

    for _ in range(warmup):
        with torch.no_grad():
            _ = _forward_only()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = _forward_only()
    torch.cuda.synchronize()
    fw_ms = (time.perf_counter() - start) * 1000.0 / iters

    for _ in range(warmup):
        out = _forward_only()
        out.sum().backward()
        a.grad = None
        b.grad = None
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = _forward_only()
        out.sum().backward()
        a.grad = None
        b.grad = None
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    bw_ms = total_ms - fw_ms

    fw_ok = None
    bw_ok = None
    if check_correctness and ref_out is not None and ref_da is not None and ref_db is not None:
        a.grad = None
        b.grad = None
        out_check = _forward_only()
        out_check.sum().backward()
        try:
            torch.testing.assert_close(out_check, ref_out, rtol=1e-2, atol=1e-2)
            fw_ok = True
        except Exception:
            fw_ok = False
        try:
            torch.testing.assert_close(a.grad, ref_da, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(b.grad, ref_db, rtol=1e-2, atol=1e-2)
            bw_ok = True
        except Exception:
            bw_ok = False
        a.grad = None
        b.grad = None

    return {
        "name": name,
        "fw_ms": fw_ms,
        "bw_ms": bw_ms,
        "total_ms": total_ms,
        "fw_ok": fw_ok,
        "bw_ok": bw_ok,
    }


def _run_param_set(params: dict[str, object]) -> None:
    from steptronoss.model.utils import moe_utils as _moe_utils

    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires GPU.")
        return

    device = torch.device("cuda")
    dtype_name = params["dtype"]
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_name]

    batch_sizes = torch.full((params["group_size"],), params["batch_size"], device=device, dtype=torch.int64)
    a = torch.randn(batch_sizes.sum().item(), params["k"], device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(params["group_size"], params["k"], params["n"], device=device, dtype=dtype, requires_grad=True)

    ref_out = None
    ref_da = None
    ref_db = None
    if params["check"]:
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)
        ref_out = _baseline_grouped_gemm(a_ref, b_ref, batch_sizes)
        ref_out.sum().backward()
        ref_da = a_ref.grad.detach().clone()
        ref_db = b_ref.grad.detach().clone()

    target = "steptronoss.model.utils.moe_utils.grouped_gemm"
    if target not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {target}")

    def runner(function_imple: str | None) -> dict[str, object]:
        if function_imple == "nv_grouped_gemm" and dtype != torch.bfloat16:
            raise RuntimeError("nv_grouped_gemm requires bf16 inputs")
        use_batch_sizes = batch_sizes.cpu() if function_imple == "nv_grouped_gemm" else batch_sizes
        return _bench_variant(
            function_imple or "baseline",
            a,
            b,
            use_batch_sizes,
            function_imple,
            params["warmup"],
            params["iters"],
            params["check"],
            ref_out,
            ref_da,
            ref_db,
        )

    results = run_with_backends(target, runner)

    base = next((r for r in results if r.name == "baseline" and r.ok), None)
    print(
        f"[grouped_gemm] group={params['group_size']} batch={params['batch_size']} "
        f"k={params['k']} n={params['n']} dtype={dtype_name}"
    )
    if params["check"]:
        print("name, fw_ms, bw_ms, total_ms, speedup_vs_base, fw_ok, bw_ok")
    else:
        print("name, fw_ms, bw_ms, total_ms, speedup_vs_base")
    base_ms = None if base is None else float(base.payload["total_ms"])
    for r in results:
        if not r.ok:
            print(f"{r.name}, ERROR, {r.error}")
            continue
        payload = r.payload or {}
        speedup = (base_ms / payload["total_ms"]) if base_ms else 0.0
        if params["check"]:
            print(
                f"{r.name}, {payload['fw_ms']:.3f}, {payload['bw_ms']:.3f}, {payload['total_ms']:.3f}, "
                f"{speedup:.2f}x, {payload['fw_ok']}, {payload['bw_ok']}"
            )
        else:
            print(
                f"{r.name}, {payload['fw_ms']:.3f}, {payload['bw_ms']:.3f}, {payload['total_ms']:.3f}, {speedup:.2f}x"
            )


def main() -> int:
    for params in PARAM_SETS:
        _run_param_set(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
