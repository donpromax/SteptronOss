import functools
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from steptronoss.model.utils.moe_utils import routed_grouped_ffn
from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER, set_optimization

PARAM_SETS = [
    {
        "token_num": 8192,
        "hidden": 4096,
        "topk": 2,
        "num_experts": 36,
        "ffn_hidden": 1280,
        "warmup": 20,
        "iters": 50,
        "invalid_frac": 0.1,
        "allow_dupe": False,
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
) -> torch.Tensor:
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device, dtype=torch.int64)
    if topk > 1 and not allow_dupe:
        for i in range(1, topk):
            clash = topk_ids[:, i] == topk_ids[:, 0]
            topk_ids[clash, i] = (topk_ids[clash, i] + i) % num_experts
    if invalid_frac > 0:
        mask = torch.rand((token_num, topk), device=device) < invalid_frac
        topk_ids = topk_ids.masked_fill(mask, -1)
    return topk_ids


def _swiglu(x, swiglu_limit=None):
    left, right = torch.chunk(x, 2, dim=-1)
    left = torch.nn.functional.silu(left)
    if swiglu_limit is not None:
        left = left.clamp(max=swiglu_limit)
        right = right.clamp(min=-swiglu_limit, max=swiglu_limit)
    return left * right


def _bench_once(
    w1: torch.Tensor,
    w2: torch.Tensor,
    x: torch.Tensor,
    token_expert_ids: torch.Tensor,
    token_weights: torch.Tensor,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    act = functools.partial(_swiglu, swiglu_limit=None)
    for _ in range(warmup):
        with torch.no_grad():
            _ = routed_grouped_ffn(
                w1,
                w2,
                act,
                x,
                token_expert_ids,
                token_weights,
            )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = routed_grouped_ffn(
                w1,
                w2,
                act,
                x,
                token_expert_ids,
                token_weights,
            )
    torch.cuda.synchronize()
    fw_ms = (time.perf_counter() - start) * 1000.0 / iters

    x_run = x.detach().clone().requires_grad_(True)
    w1_run = w1.detach().clone().requires_grad_(True)
    w2_run = w2.detach().clone().requires_grad_(True)
    token_weights_run = token_weights.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        out = routed_grouped_ffn(
            w1_run,
            w2_run,
            act,
            x_run,
            token_expert_ids,
            token_weights_run,
        )
        out.sum().backward()
        x_run.grad = None
        w1_run.grad = None
        w2_run.grad = None
        token_weights_run.grad = None
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = routed_grouped_ffn(
            w1_run,
            w2_run,
            act,
            x_run,
            token_expert_ids,
            token_weights_run,
        )
        out.sum().backward()
        x_run.grad = None
        w1_run.grad = None
        w2_run.grad = None
        token_weights_run.grad = None
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0 / iters
    return {"fw_ms": fw_ms, "bw_ms": total_ms - fw_ms, "total_ms": total_ms}


def _run_param_set(params: dict[str, object]) -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires GPU.")
        return

    target = "steptronoss.model.utils.moe_utils.routed_grouped_ffn"
    if target not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {target}")

    device = torch.device("cuda")
    x = torch.randn((int(params["token_num"]), int(params["hidden"])), device=device, dtype=torch.bfloat16)
    token_expert_ids = _make_topk_ids(
        token_num=int(params["token_num"]),
        topk=int(params["topk"]),
        num_experts=int(params["num_experts"]),
        device=device,
        allow_dupe=bool(params["allow_dupe"]),
        invalid_frac=float(params["invalid_frac"]),
    )
    token_weights = torch.rand((int(params["token_num"]), int(params["topk"])), device=device, dtype=torch.float32)
    token_weights = token_weights / token_weights.sum(dim=1, keepdim=True)

    set_optimization(
        histogram="triton", index_compute="triton", moe_weighted_gather="triton", grouped_gemm="nv_grouped_gemm"
    )

    w1 = torch.randn(
        (int(params["num_experts"]), 2 * int(params["ffn_hidden"]), int(params["hidden"])),
        device=device,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        (int(params["num_experts"]), int(params["hidden"]), int(params["ffn_hidden"])),
        device=device,
        dtype=torch.bfloat16,
    )

    ref = None
    if params.get("check", False):
        set_optimization(
            histogram="triton",
            index_compute="triton",
            moe_weighted_gather="triton",
            grouped_gemm="nv_grouped_gemm",
            routed_grouped_ffn=None,
        )
        ref = routed_grouped_ffn(
            w1,
            w2,
            functools.partial(_swiglu, swiglu_limit=None),
            x,
            token_expert_ids,
            token_weights,
        )

    backends = [None, "fused"]
    results: list[tuple[str, float]] = []
    print(
        f"[routed_grouped_ffn] token_num={params['token_num']} hidden={params['hidden']} "
        f"topk={params['topk']} experts={params['num_experts']} invalid_frac={params['invalid_frac']}"
    )
    print("name, fw_ms, bw_ms, total_ms, speedup_vs_base")
    base_ms = None
    for backend in backends:
        name = backend or "baseline"
        set_optimization(
            histogram="triton",
            index_compute="triton",
            moe_weighted_gather="triton",
            grouped_gemm="nv_grouped_gemm",
            routed_grouped_ffn=backend,
        )
        if ref is not None:
            x_check = x.detach().clone().requires_grad_(True)
            w1_check = w1.detach().clone().requires_grad_(True)
            w2_check = w2.detach().clone().requires_grad_(True)
            token_weights_check = token_weights.detach().clone().requires_grad_(True)
            out = routed_grouped_ffn(
                w1_check,
                w2_check,
                functools.partial(_swiglu, swiglu_limit=None),
                x_check,
                token_expert_ids,
                token_weights_check,
            )
            out.sum().backward()
            ref_x = x.detach().clone().requires_grad_(True)
            ref_w1 = w1.detach().clone().requires_grad_(True)
            ref_w2 = w2.detach().clone().requires_grad_(True)
            ref_token_weights = token_weights.detach().clone().requires_grad_(True)
            set_optimization(
                histogram="triton",
                index_compute="triton",
                moe_weighted_gather="triton",
                grouped_gemm="nv_grouped_gemm",
                routed_grouped_ffn=None,
            )
            ref_out = routed_grouped_ffn(
                ref_w1,
                ref_w2,
                functools.partial(_swiglu, swiglu_limit=None),
                ref_x,
                token_expert_ids,
                ref_token_weights,
            )
            ref_out.sum().backward()
            set_optimization(
                histogram="triton",
                index_compute="triton",
                moe_weighted_gather="triton",
                grouped_gemm="nv_grouped_gemm",
                routed_grouped_ffn=backend,
            )
            torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(x_check.grad, ref_x.grad, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(w1_check.grad, ref_w1.grad, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(w2_check.grad, ref_w2.grad, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(token_weights_check.grad, ref_token_weights.grad, rtol=1e-2, atol=1e-2)
        payload = _bench_once(
            w1,
            w2,
            x,
            token_expert_ids,
            token_weights,
            int(params["warmup"]),
            int(params["iters"]),
        )
        fw_ms = float(payload["fw_ms"])
        bw_ms = float(payload["bw_ms"])
        total_ms = float(payload["total_ms"])
        if name == "baseline":
            base_ms = total_ms
        speedup = (base_ms / total_ms) if base_ms else 0.0
        results.append((name, total_ms))
        print(f"{name}, {fw_ms:.3f}, {bw_ms:.3f}, {total_ms:.3f}, {speedup:.2f}x")


def main() -> int:
    for params in PARAM_SETS:
        _run_param_set(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
