import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from benchmarks.common import run_with_backends
from steptronoss.model.utils.moe_utils import histogram, index_compute, moe_scatter
from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER

PARAM_SETS = [
    {
        "token_num": 8192,
        "hidden": 256,
        "topk": 2,
        "num_experts": 8,
        "warmup": 10,
        "iters": 50,
        "check": True,
    },
]

EDGE_CASES = [
    {"name": "zero_tokens", "token_num": 0, "hidden": 8, "topk": 1, "num_experts": 4},
    {"name": "all_invalid", "token_num": 8, "hidden": 8, "topk": 2, "num_experts": 4, "all_invalid": True},
    {"name": "topk1", "token_num": 16, "hidden": 8, "topk": 1, "num_experts": 4},
    {"name": "dupe_indices", "token_num": 16, "hidden": 8, "topk": 2, "num_experts": 4, "allow_dupe": True},
]


def _make_inputs(
    token_num: int,
    hidden: int,
    topk: int,
    num_experts: int,
    device: torch.device,
    allow_dupe: bool = False,
):
    x = torch.randn((token_num, hidden), device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device, dtype=torch.int64)
    if topk > 1 and not allow_dupe:
        for i in range(1, topk):
            clash = topk_ids[:, i] == topk_ids[:, 0]
            topk_ids[clash, i] = (topk_ids[clash, i] + i) % num_experts

    experts_hist = histogram(topk_ids, num_experts)
    scatter_index = index_compute(topk_ids, experts_hist)
    return x, scatter_index


def _make_inputs_from_topk(
    token_num: int,
    hidden: int,
    topk: int,
    num_experts: int,
    device: torch.device,
    allow_dupe: bool = False,
    invalid_frac: float = 0.0,
):
    x = torch.randn((token_num, hidden), device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device, dtype=torch.int64)
    if topk > 1 and not allow_dupe:
        for i in range(1, topk):
            clash = topk_ids[:, i] == topk_ids[:, 0]
            topk_ids[clash, i] = (topk_ids[clash, i] + i) % num_experts
    if invalid_frac > 0:
        mask = torch.rand_like(topk_ids.float()) < invalid_frac
        topk_ids = topk_ids.masked_fill(mask, -1)
    experts_hist = histogram(topk_ids, num_experts)
    scatter_index = index_compute(topk_ids, experts_hist)
    return x, scatter_index


def _make_edge_inputs(case: dict[str, object], device: torch.device):
    token_num = int(case["token_num"])
    hidden = int(case["hidden"])
    topk = int(case["topk"])
    num_experts = int(case["num_experts"])
    x = torch.randn((token_num, hidden), device=device, dtype=torch.bfloat16)
    if case.get("all_invalid", False):
        idx = torch.full((token_num, topk), -1, device=device, dtype=torch.int64)
        return x, idx
    x_in, idx = _make_inputs(
        token_num,
        hidden,
        topk,
        num_experts,
        device,
        allow_dupe=case.get("allow_dupe", False),
    )
    return x_in, idx


def _check_correctness_case(name: str, x: torch.Tensor, idx: torch.Tensor) -> None:
    from steptronoss.utils.optimizable import set_optimization

    target = "steptronoss.model.utils.moe_utils.moe_scatter"
    if target not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {target}")

    set_optimization(**{"moe_scatter": None})
    x_ref = x.detach().clone().requires_grad_(True)
    idx_ref = idx.detach().clone()
    ref_out = moe_scatter(x_ref, idx_ref)
    ref_out.sum().backward()
    ref_dx = x_ref.grad.detach().clone()

    def runner(_backend: str | None) -> dict[str, object]:
        x_run = x.detach().clone().requires_grad_(True)
        out = moe_scatter(x_run, idx)
        out.sum().backward()
        if not torch.isfinite(out).all():
            raise RuntimeError("non-finite output")
        if x_run.grad is not None and not torch.isfinite(x_run.grad).all():
            raise RuntimeError("non-finite grad_input")
        torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(x_run.grad, ref_dx, rtol=1e-2, atol=1e-2)
        return {"ok": True}

    results = run_with_backends(target, runner)
    for res in results:
        if res.ok:
            continue
        print(f"[moe_scatter][check] {name} {res.name}: ERROR {res.error}")


def _run(x: torch.Tensor, idx: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        _ = moe_scatter(x, idx)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        _ = moe_scatter(x, idx)
    torch.cuda.synchronize()
    fw_ms = (time.perf_counter() - start) * 1000.0 / iters
    fw_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    for _ in range(warmup):
        out = moe_scatter(x, idx)
        out.sum().backward()
        x.grad = None
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        out = moe_scatter(x, idx)
        out.sum().backward()
        x.grad = None
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    total_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    bw_ms = total_ms - fw_ms

    return {
        "fw_ms": fw_ms,
        "bw_ms": bw_ms,
        "total_ms": total_ms,
        "fw_peak_mb": fw_peak_mb,
        "total_peak_mb": total_peak_mb,
    }


def _run_param_set(params: dict[str, object]) -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires GPU.")
        return

    device = torch.device("cuda")
    if params.get("check", False):
        for case in EDGE_CASES:
            x_case, idx_case = _make_edge_inputs(case, device)
            _check_correctness_case(case["name"], x_case, idx_case)
        # randomized stress: invalids + duplicates
        torch.manual_seed(0)
        for i in range(5):
            x_case, idx_case = _make_inputs_from_topk(
                token_num=256,
                hidden=64,
                topk=2,
                num_experts=8,
                device=device,
                allow_dupe=True,
                invalid_frac=0.3 if i % 2 == 0 else 0.0,
            )
            _check_correctness_case(f"stress_{i}", x_case, idx_case)

    x, idx = _make_inputs(params["token_num"], params["hidden"], params["topk"], params["num_experts"], device)
    x.requires_grad_(True)

    target = "steptronoss.model.utils.moe_utils.moe_scatter"
    if target not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {target}")

    def runner(_backend: str | None) -> dict[str, object]:
        return _run(x, idx, params["warmup"], params["iters"])

    results = run_with_backends(target, runner)
    print(
        f"[moe_scatter] token_num={params['token_num']} hidden={params['hidden']} topk={params['topk']} "
        f"experts={params['num_experts']}"
    )
    base_ms = None
    for res in results:
        if not res.ok:
            print(f"[moe_scatter] {res.name}: ERROR {res.error}")
            continue
        fw_ms = float(res.payload["fw_ms"])
        bw_ms = float(res.payload["bw_ms"])
        total_ms = float(res.payload["total_ms"])
        fw_peak = float(res.payload["fw_peak_mb"])
        total_peak = float(res.payload["total_peak_mb"])
        if res.name == "baseline":
            base_ms = total_ms
        speedup = (base_ms / total_ms) if base_ms else 0.0
        print(
            f"[moe_scatter] {res.name}: fw={fw_ms:.3f} ms, bw={bw_ms:.3f} ms, total={total_ms:.3f} ms, "
            f"{speedup:.2f}x, fw_peak={fw_peak:.1f}MB, total_peak={total_peak:.1f}MB"
        )


def main() -> int:
    for params in PARAM_SETS:
        _run_param_set(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
