import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from benckmarks.common import run_with_backends
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
    },
]


def _make_inputs(token_num: int, hidden: int, topk: int, num_experts: int, device: torch.device):
    x = torch.randn((token_num, hidden), device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device, dtype=torch.int64)
    if topk > 1:
        for i in range(1, topk):
            clash = topk_ids[:, i] == topk_ids[:, 0]
            topk_ids[clash, i] = (topk_ids[clash, i] + i) % num_experts

    experts_hist = histogram(topk_ids, num_experts)
    scatter_index = index_compute(topk_ids, experts_hist)
    return x, scatter_index


def _run(x: torch.Tensor, idx: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = moe_scatter(x, idx)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = moe_scatter(x, idx)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def _run_param_set(params: dict[str, object]) -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires GPU.")
        return

    device = torch.device("cuda")
    x, idx = _make_inputs(params["token_num"], params["hidden"], params["topk"], params["num_experts"], device)

    target = "steptronoss.model.utils.moe_utils.moe_scatter"
    if target not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {target}")

    def runner(_backend: str | None) -> dict[str, object]:
        ms = _run(x, idx, params["warmup"], params["iters"])
        tokens_per_iter = params["token_num"] * params["topk"]
        tput = tokens_per_iter / (ms / 1000.0)
        return {"ms": ms, "tput": tput}

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
        ms = float(res.payload["ms"])
        tput = float(res.payload["tput"])
        if res.name == "baseline":
            base_ms = ms
        speedup = (base_ms / ms) if base_ms else 0.0
        print(f"[moe_scatter] {res.name}: {ms:.3f} ms/iter, {tput:,.0f} tokens/s, {speedup:.2f}x")


def main() -> int:
    for params in PARAM_SETS:
        _run_param_set(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
