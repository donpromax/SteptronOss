import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from benckmarks.common import run_with_backends
from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER

PARAM_SETS = [
    {
        "token_num": 8192,
        "hidden": 256,
        "topk": 2,
        "num_experts": 8,
        "invalid_frac": 0.1,
        "warmup": 10,
        "iters": 50,
        "check": True,
    },
]


def _make_inputs(params: dict[str, object], device: torch.device):
    token_num = params["token_num"]
    hidden = params["hidden"]
    topk = params["topk"]

    x = torch.randn((token_num * topk, hidden), device=device, dtype=torch.bfloat16)

    topk_ids = torch.randint(0, token_num * topk, (token_num, topk), device=device, dtype=torch.int64)
    if params["invalid_frac"] > 0:
        mask = torch.rand_like(topk_ids.float()) < params["invalid_frac"]
        topk_ids[mask] = -1

    weights = torch.rand((token_num, topk), device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)

    return x, topk_ids, weights


def _bench_variant(
    x: torch.Tensor,
    idx: torch.Tensor,
    w: torch.Tensor,
    warmup: int,
    iters: int,
    check_correctness: bool,
    ref_out: torch.Tensor | None,
    ref_dx: torch.Tensor | None,
    ref_dw: torch.Tensor | None,
):
    def _forward_only():
        from steptronoss.model.utils.moe_utils import moe_weighted_gather

        return moe_weighted_gather(x, idx, w)

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
        x.grad = None
        w.grad = None
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = _forward_only()
        out.sum().backward()
        x.grad = None
        w.grad = None
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    bw_ms = total_ms - fw_ms

    fw_ok = None
    bw_ok = None
    if check_correctness and ref_out is not None and ref_dx is not None and ref_dw is not None:
        x.grad = None
        w.grad = None
        out_check = _forward_only()
        out_check.sum().backward()
        try:
            torch.testing.assert_close(out_check, ref_out, rtol=1e-2, atol=1e-2)
            fw_ok = True
        except Exception:
            fw_ok = False
        try:
            torch.testing.assert_close(x.grad, ref_dx, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(w.grad, ref_dw, rtol=1e-2, atol=1e-2)
            bw_ok = True
        except Exception:
            bw_ok = False
        x.grad = None
        w.grad = None

    return {
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
    x, idx, w = _make_inputs(params, device)
    x.requires_grad_(True)
    w.requires_grad_(True)

    ref_out = None
    ref_dx = None
    ref_dw = None
    if params["check"]:
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(**{"moe_weighted_gather": None})
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = w.detach().clone().requires_grad_(True)
        from steptronoss.model.utils.moe_utils import moe_weighted_gather

        ref_out = moe_weighted_gather(x_ref, idx, w_ref)
        ref_out.sum().backward()
        ref_dx = x_ref.grad.detach().clone()
        ref_dw = w_ref.grad.detach().clone()

    target = "steptronoss.model.utils.moe_utils.moe_weighted_gather"
    if target not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {target}")

    def runner(_backend: str | None) -> dict[str, object]:
        return _bench_variant(
            x,
            idx,
            w,
            params["warmup"],
            params["iters"],
            params["check"],
            ref_out,
            ref_dx,
            ref_dw,
        )

    results = run_with_backends(target, runner)
    print(
        f"[moe_gather] token_num={params['token_num']} hidden={params['hidden']} topk={params['topk']} "
        f"experts={params['num_experts']} invalid_frac={params['invalid_frac']}"
    )
    base = next((r for r in results if r.name == "baseline" and r.ok), None)
    base_ms = None if base is None else float(base.payload["total_ms"])
    if params["check"]:
        print("name, fw_ms, bw_ms, total_ms, speedup_vs_base, fw_ok, bw_ok")
    else:
        print("name, fw_ms, bw_ms, total_ms, speedup_vs_base")
    for res in results:
        if not res.ok:
            print(f"[moe_gather] {res.name}: ERROR {res.error}")
            continue
        payload = res.payload or {}
        speedup = (base_ms / payload["total_ms"]) if base_ms else 0.0
        if params["check"]:
            print(
                f"{res.name}, {payload['fw_ms']:.3f}, {payload['bw_ms']:.3f}, {payload['total_ms']:.3f}, "
                f"{speedup:.2f}x, {payload['fw_ok']}, {payload['bw_ok']}"
            )
        else:
            print(
                f"{res.name}, {payload['fw_ms']:.3f}, {payload['bw_ms']:.3f}, {payload['total_ms']:.3f}, {speedup:.2f}x"
            )


def main() -> int:
    for params in PARAM_SETS:
        _run_param_set(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
