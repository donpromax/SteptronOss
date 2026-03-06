import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from benchmarks.common import run_with_backends
from steptronoss.model.utils.moe_utils import histogram, index_compute
from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER, set_optimization

PARAM_SETS = [
    {
        "token_num": 8192,
        "topk": 2,
        "num_experts": 128,
        "warmup": 20,
        "iters": 100,
        "invalid_frac": 0.1,
        "allow_dupe": False,
        "check": True,
    },
    {
        "token_num": 32768,
        "topk": 2,
        "num_experts": 288,
        "warmup": 20,
        "iters": 100,
        "invalid_frac": 0.0,
        "allow_dupe": True,
        "check": True,
    },
]


def _make_topk_ids(
    token_num: int,
    topk: int,
    num_experts: int,
    device: torch.device,
    *,
    allow_dupe: bool,
    invalid_frac: float,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device, dtype=torch.int64)
    if topk > 1 and not allow_dupe:
        for i in range(1, topk):
            clash = topk_ids[:, i] == topk_ids[:, 0]
            topk_ids[clash, i] = (topk_ids[clash, i] + i) % num_experts
    if invalid_frac > 0:
        mask = torch.rand((token_num, topk), device=device) < invalid_frac
        topk_ids = topk_ids.masked_fill(mask, -1)
    return topk_ids.to(dtype)


def _bench_histogram(topk_ids: torch.Tensor, num_experts: int, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        _ = histogram(topk_ids, expert_num=num_experts)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = histogram(topk_ids, expert_num=num_experts)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    return {"total_ms": total_ms}


def _bench_index_compute(
    topk_ids: torch.Tensor,
    experts_hist: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        _ = index_compute(topk_ids, experts_hist)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = index_compute(topk_ids, experts_hist)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    return {"total_ms": total_ms}


def _run_histogram(params: dict[str, object], topk_ids: torch.Tensor) -> None:
    target = "steptronoss.model.utils.moe_utils.histogram"
    if target not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {target}")

    ref = None
    if params.get("check", False):
        set_optimization(histogram=None)
        ref = histogram(topk_ids, expert_num=int(params["num_experts"]))

    def runner(_backend: str | None) -> dict[str, object]:
        out = histogram(topk_ids, expert_num=int(params["num_experts"]))
        if ref is not None:
            torch.testing.assert_close(out, ref)
        bench = _bench_histogram(topk_ids, int(params["num_experts"]), int(params["warmup"]), int(params["iters"]))
        return {**bench}

    results = run_with_backends(target, runner)
    base_ms = None
    print(
        f"[moe_routing][histogram] token_num={params['token_num']} topk={params['topk']} "
        f"experts={params['num_experts']} invalid_frac={params['invalid_frac']} allow_dupe={params['allow_dupe']}"
    )
    print("name, total_ms, speedup_vs_base")
    for res in results:
        if not res.ok:
            print(f"{res.name}, ERROR, {res.error}")
            continue
        total_ms = float(res.payload["total_ms"])
        if res.name == "baseline":
            base_ms = total_ms
        speedup = (base_ms / total_ms) if base_ms else 0.0
        print(f"{res.name}, {total_ms:.3f}, {speedup:.2f}x")


def _run_index_compute(params: dict[str, object], topk_ids: torch.Tensor) -> None:
    target = "steptronoss.model.utils.moe_utils.index_compute"
    if target not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {target}")

    set_optimization(histogram=None, index_compute=None)
    experts_hist = histogram(topk_ids, expert_num=int(params["num_experts"]))
    ref = None
    if params.get("check", False):
        ref = index_compute(topk_ids, experts_hist)

    def runner(_backend: str | None) -> dict[str, object]:
        out = index_compute(topk_ids, experts_hist)
        if ref is not None:
            torch.testing.assert_close(out, ref)
        bench = _bench_index_compute(topk_ids, experts_hist, int(params["warmup"]), int(params["iters"]))
        return {**bench}

    results = run_with_backends(target, runner)
    base_ms = None
    print(
        f"[moe_routing][index_compute] token_num={params['token_num']} topk={params['topk']} "
        f"experts={params['num_experts']} invalid_frac={params['invalid_frac']} allow_dupe={params['allow_dupe']}"
    )
    print("name, total_ms, speedup_vs_base")
    for res in results:
        if not res.ok:
            print(f"{res.name}, ERROR, {res.error}")
            continue
        total_ms = float(res.payload["total_ms"])
        if res.name == "baseline":
            base_ms = total_ms
        speedup = (base_ms / total_ms) if base_ms else 0.0
        print(f"{res.name}, {total_ms:.3f}, {speedup:.2f}x")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires GPU.")
        return 0

    device = torch.device("cuda")
    for params in PARAM_SETS:
        topk_ids = _make_topk_ids(
            token_num=int(params["token_num"]),
            topk=int(params["topk"]),
            num_experts=int(params["num_experts"]),
            device=device,
            allow_dupe=bool(params["allow_dupe"]),
            invalid_frac=float(params["invalid_frac"]),
        )
        _run_histogram(params, topk_ids)
        _run_index_compute(params, topk_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
