import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from steptronoss.core.parallel_state import PM
from steptronoss.core.tensor_parallel import vocab_parallel_cross_entropy
from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER, set_optimization

TARGET = "steptronoss.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy"
TARGET_NAME = TARGET.split(".")[-1]
PREFERRED_BACKENDS = ("triton", "torch_compile")

PARAM_SETS = [
    {
        "tp_size": 2,
        "seq_len": 1024,
        "batch_size": 4,
        "vocab_per_partition": 16384,
        "label_smoothing": 0.0,
        "warmup": 5,
        "iters": 20,
        "check": True,
    },
    {
        "tp_size": 2,
        "seq_len": 512,
        "batch_size": 4,
        "vocab_per_partition": 32768,
        "label_smoothing": 0.1,
        "warmup": 5,
        "iters": 20,
        "check": True,
    },
]


def _available_backends() -> list[str | None]:
    if TARGET not in OPTIMIZABLE_REGISTER:
        raise RuntimeError(f"Target not registered: {TARGET}")

    alternatives = OPTIMIZABLE_REGISTER[TARGET]["alternatives"]
    ordered = [None]
    ordered.extend(name for name in PREFERRED_BACKENDS if alternatives.get(name) is not None)
    ordered.extend(
        sorted(name for name, func in alternatives.items() if func is not None and name not in PREFERRED_BACKENDS)
    )
    return ordered


def _clear_parallel_state() -> None:
    PM._all_groups.clear()
    PM.parallels.clear()
    PM.all_parallels.clear()
    PM._stack.clear()
    PM._cur_cfg = None
    PM._rng_seeds.clear()
    PM.rng_states.clear()


def _init_dist_and_mesh(tp_size: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())

    did_init = False
    if not dist.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            raise RuntimeError(
                f"Benchmark requires torchrun. Example: `torchrun --nproc-per-node={tp_size} benchmarks/benchmark_cross_entropy.py`"
            )
        dist.init_process_group(backend="nccl")
        did_init = True

    if dist.get_backend() != "nccl":
        raise RuntimeError("Benchmark requires NCCL backend")
    if dist.get_world_size() != tp_size:
        raise RuntimeError(f"Benchmark assumes WORLD_SIZE={tp_size}, got {dist.get_world_size()}")

    PM.initialize(backend="nccl")
    parallel_cfg = PM._cur_cfg or None
    needs_mesh = (
        parallel_cfg is None
        or getattr(parallel_cfg, "tensor_model_parallel_size", None) != tp_size
        or getattr(parallel_cfg, "pipeline_model_parallel_size", None) != 1
        or getattr(parallel_cfg, "context_parallel_size", None) != 1
        or getattr(parallel_cfg, "expert_model_parallel_size", None) != 1
        or getattr(parallel_cfg, "expert_tensor_parallel_size", None) != 1
        or getattr(parallel_cfg, "virtual_pipeline_model_parallel_size", None) != 1
    )
    if needs_mesh:
        parallel_cfg = ParallelConfig()
        parallel_cfg.tensor_model_parallel_size = tp_size
        parallel_cfg.pipeline_model_parallel_size = 1
        parallel_cfg.context_parallel_size = 1
        parallel_cfg.expert_model_parallel_size = 1
        parallel_cfg.expert_tensor_parallel_size = 1
        parallel_cfg.virtual_pipeline_model_parallel_size = 1
        PM.set_mesh(parallel_cfg)

    return did_init


def _cleanup_dist(did_init: bool) -> None:
    if did_init and dist.is_initialized():
        dist.destroy_process_group()
    _clear_parallel_state()


def _make_inputs(params: dict[str, object], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(params["seq_len"])
    batch_size = int(params["batch_size"])
    vocab_per_partition = int(params["vocab_per_partition"])
    tp_size = int(params["tp_size"])

    logits_seed = 1234 + PM.rank_in("TP")
    logits_gen = torch.Generator()
    logits_gen.manual_seed(logits_seed)
    base_logits = torch.randn(
        (seq_len, batch_size, vocab_per_partition),
        generator=logits_gen,
        dtype=torch.float32,
    ).to(device)

    target_gen = torch.Generator()
    target_gen.manual_seed(4321)
    target = torch.randint(
        0,
        vocab_per_partition * tp_size,
        (seq_len, batch_size),
        generator=target_gen,
        dtype=torch.int64,
    ).to(device)
    return base_logits, target


def _prepare_backward_logits(base_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits_leaf = base_logits.clone().detach().requires_grad_(True)
    logits_run = logits_leaf + 0.0
    return logits_leaf, logits_run


def _sync_tp_max(value: float) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=PM.group_of("TP"))
    return float(tensor.item())


def _check_correctness(base_logits: torch.Tensor, target: torch.Tensor, label_smoothing: float) -> None:
    set_optimization(**{TARGET_NAME: None})
    ref_logits_leaf, ref_logits_run = _prepare_backward_logits(base_logits)
    ref_losses, ref_acc = vocab_parallel_cross_entropy(ref_logits_run, target, label_smoothing)
    ref_loss = ref_losses.mean()
    ref_loss.backward()
    ref_loss_tensor = ref_losses.detach().clone()
    ref_acc_tensor = ref_acc.detach().clone()
    ref_grad = ref_logits_leaf.grad.detach().clone()

    for backend in _available_backends():
        name = backend or "baseline"
        set_optimization(**{TARGET_NAME: backend})
        logits_leaf, logits_run = _prepare_backward_logits(base_logits)
        losses, acc = vocab_parallel_cross_entropy(logits_run, target, label_smoothing)
        loss = losses.mean()
        loss.backward()

        if not torch.isfinite(losses).all():
            raise RuntimeError(f"{name}: non-finite losses")
        if not torch.isfinite(acc):
            raise RuntimeError(f"{name}: non-finite acc")
        if logits_leaf.grad is None or not torch.isfinite(logits_leaf.grad).all():
            raise RuntimeError(f"{name}: non-finite grad_input")

        torch.testing.assert_close(losses, ref_loss_tensor, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(acc, ref_acc_tensor, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(logits_leaf.grad, ref_grad, rtol=1e-4, atol=1e-4)

    set_optimization(**{TARGET_NAME: None})


def _benchmark_variant(
    base_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    tp_group = PM.group_of("TP")

    dist.barrier(group=tp_group)
    for _ in range(warmup):
        logits = base_logits.clone()
        with torch.no_grad():
            _ = vocab_parallel_cross_entropy(logits, target, label_smoothing)
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)

    torch.cuda.reset_peak_memory_stats()
    fw_starts: list[torch.cuda.Event] = []
    fw_ends: list[torch.cuda.Event] = []
    for _ in range(iters):
        logits = base_logits.clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = vocab_parallel_cross_entropy(logits, target, label_smoothing)
        end.record()
        fw_starts.append(start)
        fw_ends.append(end)
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)
    fw_ms = sum(start.elapsed_time(end) for start, end in zip(fw_starts, fw_ends)) / iters
    fw_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    dist.barrier(group=tp_group)
    for _ in range(warmup):
        logits_leaf, logits_run = _prepare_backward_logits(base_logits)
        losses, _ = vocab_parallel_cross_entropy(logits_run, target, label_smoothing)
        losses.mean().backward()
        del logits_leaf
        del logits_run
        del losses
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)

    torch.cuda.reset_peak_memory_stats()
    total_starts: list[torch.cuda.Event] = []
    total_ends: list[torch.cuda.Event] = []
    for _ in range(iters):
        logits_leaf, logits_run = _prepare_backward_logits(base_logits)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        losses, _ = vocab_parallel_cross_entropy(logits_run, target, label_smoothing)
        losses.mean().backward()
        end.record()
        total_starts.append(start)
        total_ends.append(end)
        del logits_leaf
        del logits_run
        del losses
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)
    total_ms = sum(start.elapsed_time(end) for start, end in zip(total_starts, total_ends)) / iters
    total_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    fw_ms = _sync_tp_max(fw_ms)
    total_ms = _sync_tp_max(total_ms)
    fw_peak_mb = _sync_tp_max(fw_peak_mb)
    total_peak_mb = _sync_tp_max(total_peak_mb)
    return {
        "fw_ms": fw_ms,
        "bw_ms": total_ms - fw_ms,
        "total_ms": total_ms,
        "fw_peak_mb": fw_peak_mb,
        "total_peak_mb": total_peak_mb,
    }


def _run_param_set(params: dict[str, object]) -> None:
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    base_logits, target = _make_inputs(params, device)

    if params.get("check", False):
        _check_correctness(base_logits, target, float(params["label_smoothing"]))

    results: list[tuple[str, dict[str, float]]] = []
    for backend in _available_backends():
        name = backend or "baseline"
        set_optimization(**{TARGET_NAME: backend})
        payload = _benchmark_variant(
            base_logits=base_logits,
            target=target,
            label_smoothing=float(params["label_smoothing"]),
            warmup=int(params["warmup"]),
            iters=int(params["iters"]),
        )
        results.append((name, payload))

    if PM.world_rank != 0:
        return

    base_ms = next(payload["total_ms"] for name, payload in results if name == "baseline")
    tokens_per_sec_base = int(params["seq_len"]) * int(params["batch_size"]) / (base_ms / 1000.0)
    print(
        f"[cross_entropy] tp={params['tp_size']} seq={params['seq_len']} batch={params['batch_size']} "
        f"local_vocab={params['vocab_per_partition']} global_vocab={int(params['tp_size']) * int(params['vocab_per_partition'])} "
        f"label_smoothing={params['label_smoothing']} dtype=float32"
    )
    print(f"[cross_entropy] baseline tokens/s={tokens_per_sec_base:.0f}")
    print("name, fw_ms, bw_ms, total_ms, speedup_vs_base, fw_peak_mb, total_peak_mb")
    for name, payload in results:
        speedup = base_ms / payload["total_ms"] if base_ms else 0.0
        print(
            f"{name}, {payload['fw_ms']:.3f}, {payload['bw_ms']:.3f}, {payload['total_ms']:.3f}, "
            f"{speedup:.2f}x, {payload['fw_peak_mb']:.1f}, {payload['total_peak_mb']:.1f}"
        )


def main() -> int:
    if not PARAM_SETS:
        return 0
    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires GPU.")
        return 0

    tp_sizes = {int(params["tp_size"]) for params in PARAM_SETS}
    if len(tp_sizes) != 1:
        raise RuntimeError("All parameter sets must share the same tp_size for one torchrun launch.")

    did_init = False
    try:
        did_init = _init_dist_and_mesh(tp_size=tp_sizes.pop())
        for params in PARAM_SETS:
            _run_param_set(params)
    finally:
        _cleanup_dist(did_init)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
