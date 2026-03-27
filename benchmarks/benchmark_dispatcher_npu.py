"""
NPU AllToAll Dispatcher Benchmark
torchrun --nproc_per_node=8 benchmarks/benchmark_dispatcher_npu.py
"""

from steptronoss.utils.npu_patch import apply_npu_patch

apply_npu_patch()

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.model.ep_dispatcher.token_dispatcher import TokenDispatcher
from steptronoss.utils.optimizable import set_optimization

# PARAM_SETS = [
#     {
#         "ep_size": 8,
#         "num_experts": 8,
#         "seq_len": 4096,
#         "hidden": 256,
#         "topk": 2,
#         "warmup": 5,
#         "iters": 20,
#         "check": True,
#     },
# ]

PARAM_SETS = [
    {
        "ep_size": 8,
        "num_experts": 288,
        "seq_len": 4096,
        "hidden": 4096,
        "topk": 8,
        "warmup": 5,
        "iters": 20,
        "check": True,
    },
]


def _sync() -> None:
    torch.npu.synchronize()


def _reset_peak_memory_stats() -> None:
    if hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()
    if hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()


def _peak_memory_stats_mb() -> tuple[float | None, float | None]:
    peak_allocated = None
    peak_reserved = None
    if hasattr(torch.npu, "max_memory_allocated"):
        peak_allocated = float(torch.npu.max_memory_allocated()) / (1024**2)
    if hasattr(torch.npu, "max_memory_reserved"):
        peak_reserved = float(torch.npu.max_memory_reserved()) / (1024**2)
    return peak_allocated, peak_reserved


def _measure_host_memory(fn):
    proc = psutil.Process()
    stop = threading.Event()
    peak_rss_mb = 0.0
    peak_host_used_mb = 0.0
    peak_host_percent = 0.0

    def poll():
        nonlocal peak_rss_mb, peak_host_used_mb, peak_host_percent
        while not stop.is_set():
            try:
                rss_mb = proc.memory_info().rss / (1024**2)
                vm = psutil.virtual_memory()
                used_mb = vm.used / (1024**2)
                percent = float(vm.percent)
                peak_rss_mb = max(peak_rss_mb, rss_mb)
                peak_host_used_mb = max(peak_host_used_mb, used_mb)
                peak_host_percent = max(peak_host_percent, percent)
            except Exception:
                pass
            stop.wait(0.05)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)
    return result, peak_rss_mb, peak_host_used_mb, peak_host_percent


def _init_dist_and_mesh(ep_size: int) -> bool:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is required for benchmark_dispatcher_npu.py")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    did_init = False
    if dist.is_initialized():
        if dist.get_backend() != "hccl":
            raise RuntimeError("Benchmark requires HCCL backend")
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(ep_size))
        os.environ.setdefault("LOCAL_RANK", "0")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl")
        did_init = True

    if dist.get_world_size() != ep_size:
        raise RuntimeError(f"Benchmark assumes WORLD_SIZE={ep_size}")

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    PM.initialize(backend="hccl")
    parallel_cfg = PM._cur_cfg or None
    if parallel_cfg is None or PM.size_of("EP") != ep_size:
        parallel_cfg = ParallelConfig()
        parallel_cfg.tensor_model_parallel_size = 1
        parallel_cfg.pipeline_model_parallel_size = 1
        parallel_cfg.context_parallel_size = 1
        parallel_cfg.expert_model_parallel_size = ep_size
        parallel_cfg.expert_tensor_parallel_size = 1
        parallel_cfg.virtual_pipeline_model_parallel_size = 1
        PM.set_mesh(parallel_cfg)

    return did_init


def _cleanup_dist(did_init: bool) -> None:
    if did_init and dist.is_initialized():
        dist.destroy_process_group()
        PM._all_groups.clear()
        PM.parallels.clear()
        PM.all_parallels.clear()
        PM._stack.clear()
        PM._cur_cfg = None
        PM._rng_seeds.clear()
        PM.rng_states.clear()


def _make_inputs(seq_len: int, hidden: int, topk: int, num_experts: int, device: torch.device):
    hidden_states = torch.randn((seq_len, hidden), device=device, dtype=torch.bfloat16)
    token_expert_ids = torch.randint(0, num_experts, (seq_len, topk), device=device, dtype=torch.int64)
    if topk > 1:
        for i in range(1, topk):
            clash = token_expert_ids[:, i] == token_expert_ids[:, 0]
            token_expert_ids[clash, i] = (token_expert_ids[clash, i] + i) % num_experts
    token_expert_weights = torch.rand((seq_len, topk), device=device, dtype=torch.float32)
    token_expert_weights = token_expert_weights / token_expert_weights.sum(dim=1, keepdim=True)
    return hidden_states, token_expert_ids, token_expert_weights


def _max_abs_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
        return float((x.to(torch.int64) - y.to(torch.int64)).abs().max().item())
    return float((x.float() - y.float()).abs().max().item())


def _check_close(x: torch.Tensor, y: torch.Tensor, rtol: float, atol: float) -> bool:
    try:
        torch.testing.assert_close(x, y, rtol=rtol, atol=atol)
        return True
    except Exception:
        return False


def _dispatcher_loss(output: torch.Tensor, recv_prob: torch.Tensor) -> torch.Tensor:
    # Cover both hidden-state restore and prob communication paths in backward.
    return output.float().sum() + recv_prob.float().sum()


def _run_dispatcher(
    backend: str | None,
    hidden_states: torch.Tensor,
    token_expert_ids: torch.Tensor,
    token_expert_weights: torch.Tensor,
    num_experts: int,
    warmup: int,
    iters: int,
):
    set_optimization(**{"TokenDispatcher": backend})
    dist.barrier()
    _sync()
    dist.barrier()

    dispatcher = TokenDispatcher("EP", num_experts=num_experts)
    recv_hidden = recv_idx = recv_prob = out = None
    _reset_peak_memory_stats()
    for _ in range(warmup):
        recv_hidden, recv_idx, recv_prob = dispatcher.dispatch(hidden_states, token_expert_ids, token_expert_weights)
        out = dispatcher.combine(recv_hidden)
    _sync()
    dist.barrier()

    _reset_peak_memory_stats()

    def timed_run():
        nonlocal recv_hidden, recv_idx, recv_prob, out
        start = time.perf_counter()
        for _ in range(iters):
            recv_hidden, recv_idx, recv_prob = dispatcher.dispatch(
                hidden_states, token_expert_ids, token_expert_weights
            )
            out = dispatcher.combine(recv_hidden)
        _sync()
        dist.barrier()
        return (time.perf_counter() - start) * 1000.0 / iters

    ms_per_iter, peak_rss_mb, peak_host_used_mb, peak_host_percent = _measure_host_memory(timed_run)
    peak_allocated_mb, peak_reserved_mb = _peak_memory_stats_mb()

    grad_hidden_states = hidden_states.detach().clone().requires_grad_(True)
    grad_token_expert_weights = token_expert_weights.detach().clone().requires_grad_(True)
    dispatcher_bw = TokenDispatcher("EP", num_experts=num_experts)
    recv_prob_bw = out_bw = None
    grad_hidden = grad_token_expert_weights_out = None

    for _ in range(warmup):
        recv_hidden_bw, _, recv_prob_bw = dispatcher_bw.dispatch(
            grad_hidden_states,
            token_expert_ids,
            grad_token_expert_weights,
        )
        out_bw = dispatcher_bw.combine(recv_hidden_bw)
        _dispatcher_loss(out_bw, recv_prob_bw).backward()
        grad_hidden_states.grad = None
        grad_token_expert_weights.grad = None
    _sync()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(iters):
        recv_hidden_bw, _, recv_prob_bw = dispatcher_bw.dispatch(
            grad_hidden_states,
            token_expert_ids,
            grad_token_expert_weights,
        )
        out_bw = dispatcher_bw.combine(recv_hidden_bw)
        _dispatcher_loss(out_bw, recv_prob_bw).backward()
        grad_hidden = grad_hidden_states.grad.detach().clone()
        grad_token_expert_weights_out = grad_token_expert_weights.grad.detach().clone()
        grad_hidden_states.grad = None
        grad_token_expert_weights.grad = None
    _sync()
    dist.barrier()
    total_ms_per_iter = (time.perf_counter() - start) * 1000.0 / iters

    return (
        ms_per_iter,
        total_ms_per_iter,
        out,
        recv_hidden,
        recv_idx,
        recv_prob,
        out_bw.detach().clone(),
        grad_hidden,
        grad_token_expert_weights_out,
        peak_allocated_mb,
        peak_reserved_mb,
        peak_rss_mb,
        peak_host_used_mb,
        peak_host_percent,
        bool(getattr(dispatcher, "last_dispatch_used_fused_permute", False)),
        bool(getattr(dispatcher, "last_combine_used_fused_unpermute", False)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    defaults = dict(PARAM_SETS[0])

    parser.add_argument("--ep-size", type=int, default=defaults["ep_size"])
    parser.add_argument("--num-experts", type=int, default=defaults["num_experts"])
    parser.add_argument("--seq-len", type=int, default=defaults["seq_len"])
    parser.add_argument("--hidden", type=int, default=defaults["hidden"])
    parser.add_argument("--topk", type=int, default=defaults["topk"])
    parser.add_argument("--warmup", type=int, default=defaults["warmup"])
    parser.add_argument("--iters", type=int, default=defaults["iters"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--check", action="store_true", default=defaults["check"])
    args = parser.parse_args()

    did_init = False
    try:
        did_init = _init_dist_and_mesh(args.ep_size)
        num_experts = args.num_experts
        device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
        hidden_states, token_expert_ids, token_expert_weights = _make_inputs(
            args.seq_len, args.hidden, args.topk, num_experts, device
        )

        (
            base_fw_ms,
            base_total_ms,
            base_out,
            base_recv_hidden,
            base_recv_idx,
            base_recv_prob,
            base_out_bw,
            base_grad_hidden,
            base_grad_prob,
            base_peak_alloc_mb,
            base_peak_reserved_mb,
            base_peak_rss_mb,
            base_peak_host_used_mb,
            base_peak_host_percent,
            base_fused_permute,
            base_fused_unpermute,
        ) = _run_dispatcher(
            None,
            hidden_states,
            token_expert_ids,
            token_expert_weights,
            num_experts,
            args.warmup,
            args.iters,
        )

        (
            npu_fw_ms,
            npu_total_ms,
            npu_out,
            npu_recv_hidden,
            npu_recv_idx,
            npu_recv_prob,
            npu_out_bw,
            npu_grad_hidden,
            npu_grad_prob,
            npu_peak_alloc_mb,
            npu_peak_reserved_mb,
            npu_peak_rss_mb,
            npu_peak_host_used_mb,
            npu_peak_host_percent,
            npu_fused_permute,
            npu_fused_unpermute,
        ) = _run_dispatcher(
            "npu_alltoall",
            hidden_states,
            token_expert_ids,
            token_expert_weights,
            num_experts,
            args.warmup,
            args.iters,
        )

        torch.testing.assert_close(npu_recv_hidden, base_recv_hidden, rtol=args.rtol, atol=args.atol)
        torch.testing.assert_close(npu_recv_idx, base_recv_idx, rtol=0, atol=0)
        torch.testing.assert_close(npu_recv_prob, base_recv_prob, rtol=args.rtol, atol=args.atol)
        torch.testing.assert_close(npu_out, base_out, rtol=args.rtol, atol=args.atol)
        torch.testing.assert_close(npu_out_bw, base_out_bw, rtol=args.rtol, atol=args.atol)
        torch.testing.assert_close(npu_grad_hidden, base_grad_hidden, rtol=args.rtol, atol=args.atol)
        torch.testing.assert_close(npu_grad_prob, base_grad_prob, rtol=args.rtol, atol=args.atol)

        if PM.rank_in("EP") == 0:
            total_tokens = args.seq_len * args.ep_size
            base_bw_ms = base_total_ms - base_fw_ms
            npu_bw_ms = npu_total_ms - npu_fw_ms
            base_fw_tokens_per_s = total_tokens / (base_fw_ms / 1000.0)
            base_bw_tokens_per_s = total_tokens / (base_bw_ms / 1000.0)
            base_total_tokens_per_s = total_tokens / (base_total_ms / 1000.0)
            npu_fw_tokens_per_s = total_tokens / (npu_fw_ms / 1000.0)
            npu_bw_tokens_per_s = total_tokens / (npu_bw_ms / 1000.0)
            npu_total_tokens_per_s = total_tokens / (npu_total_ms / 1000.0)
            print(
                f"[dispatcher_npu] ep={args.ep_size} num_experts={num_experts} "
                f"seq_len={args.seq_len} hidden={args.hidden} topk={args.topk}"
            )
            print(
                f"[dispatcher_npu] baseline: fw_ms={base_fw_ms:.3f}, bw_ms={base_bw_ms:.3f}, "
                f"total_ms={base_total_ms:.3f}, fw_tokens/s={base_fw_tokens_per_s:,.0f}, "
                f"bw_tokens/s={base_bw_tokens_per_s:,.0f}, total_tokens/s={base_total_tokens_per_s:,.0f}"
            )
            print(
                f"[dispatcher_npu] baseline_peak_mem: allocated_mb={base_peak_alloc_mb:.1f}, "
                f"reserved_mb={base_peak_reserved_mb:.1f}"
            )
            print(
                f"[dispatcher_npu] baseline_host_mem: process_rss_mb={base_peak_rss_mb:.1f}, "
                f"host_used_mb={base_peak_host_used_mb:.1f}, host_used_percent={base_peak_host_percent:.1f}"
            )
            print(
                f"[dispatcher_npu] baseline_fused_ops: permute={base_fused_permute}, unpermute={base_fused_unpermute}"
            )
            print(
                f"[dispatcher_npu] npu_alltoall: fw_ms={npu_fw_ms:.3f}, bw_ms={npu_bw_ms:.3f}, "
                f"total_ms={npu_total_ms:.3f}, fw_tokens/s={npu_fw_tokens_per_s:,.0f}, "
                f"bw_tokens/s={npu_bw_tokens_per_s:,.0f}, total_tokens/s={npu_total_tokens_per_s:,.0f}"
            )
            print(
                f"[dispatcher_npu] speedup_vs_baseline: fw_throughput={npu_fw_tokens_per_s / base_fw_tokens_per_s:.2f}x, "
                f"total_throughput={npu_total_tokens_per_s / base_total_tokens_per_s:.2f}x"
            )
            print(
                f"[dispatcher_npu] npu_alltoall_peak_mem: allocated_mb={npu_peak_alloc_mb:.1f}, "
                f"reserved_mb={npu_peak_reserved_mb:.1f}"
            )
            print(
                f"[dispatcher_npu] npu_alltoall_host_mem: process_rss_mb={npu_peak_rss_mb:.1f}, "
                f"host_used_mb={npu_peak_host_used_mb:.1f}, host_used_percent={npu_peak_host_percent:.1f}"
            )
            print(
                f"[dispatcher_npu] npu_alltoall_fused_ops: permute={npu_fused_permute}, unpermute={npu_fused_unpermute}"
            )
            print("metric, close, max_abs_diff")
            print(
                f"recv_hidden, {_check_close(npu_recv_hidden, base_recv_hidden, args.rtol, args.atol)}, "
                f"{_max_abs_diff(npu_recv_hidden, base_recv_hidden):.6f}"
            )
            print(
                f"recv_idx, {_check_close(npu_recv_idx, base_recv_idx, 0.0, 0.0)}, {_max_abs_diff(npu_recv_idx, base_recv_idx):.6f}"
            )
            print(
                f"recv_prob, {_check_close(npu_recv_prob, base_recv_prob, args.rtol, args.atol)}, "
                f"{_max_abs_diff(npu_recv_prob, base_recv_prob):.6f}"
            )
            print(
                f"combine_out, {_check_close(npu_out, base_out, args.rtol, args.atol)}, {_max_abs_diff(npu_out, base_out):.6f}"
            )
            print(
                f"combine_out_bw, {_check_close(npu_out_bw, base_out_bw, args.rtol, args.atol)}, "
                f"{_max_abs_diff(npu_out_bw, base_out_bw):.6f}"
            )
            print(
                f"grad_hidden, {_check_close(npu_grad_hidden, base_grad_hidden, args.rtol, args.atol)}, "
                f"{_max_abs_diff(npu_grad_hidden, base_grad_hidden):.6f}"
            )
            print(
                f"grad_prob, {_check_close(npu_grad_prob, base_grad_prob, args.rtol, args.atol)}, "
                f"{_max_abs_diff(npu_grad_prob, base_grad_prob):.6f}"
            )
        return 0
    finally:
        set_optimization(**{"TokenDispatcher": None})
        _cleanup_dist(did_init)


if __name__ == "__main__":
    raise SystemExit(main())
