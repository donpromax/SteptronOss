import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from benchmarks.common import run_with_backends
from steptronoss.core.parallel_state import PM
from steptronoss.exp.base_exp import ParallelConfig
from steptronoss.model.ep_dispatcher.token_dispatcher import TokenDispatcher
from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER

PARAM_SETS = [
    {
        "ep_size": 8,
        "num_experts": 8,
        "seq_len": 4096,
        "hidden": 256,
        "topk": 2,
        "warmup": 5,
        "iters": 20,
        "check": True,
    },
]


def _init_dist_and_mesh(ep_size: int) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    did_init = False
    if dist.is_initialized():
        if dist.get_backend() != "nccl":
            raise RuntimeError("Benchmark requires NCCL backend")
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(ep_size))
        os.environ.setdefault("LOCAL_RANK", "0")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
        did_init = True

    if dist.get_world_size() != ep_size:
        raise RuntimeError(f"Benchmark assumes WORLD_SIZE={ep_size}")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    PM.initialize(backend="nccl")
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

    # Random top-k experts per token (unique per token)
    token_expert_ids = torch.randint(0, num_experts, (seq_len, topk), device=device, dtype=torch.int64)
    if topk > 1:
        for i in range(1, topk):
            clash = token_expert_ids[:, i] == token_expert_ids[:, 0]
            token_expert_ids[clash, i] = (token_expert_ids[clash, i] + i) % num_experts

    token_expert_weights = torch.rand((seq_len, topk), device=device, dtype=torch.float32)
    token_expert_weights = token_expert_weights / token_expert_weights.sum(dim=1, keepdim=True)

    return hidden_states, token_expert_ids, token_expert_weights


def _check_deepep_alignment(hidden: int) -> None:
    hidden_bytes = hidden * 2  # bf16
    if hidden_bytes % 16 != 0 or (hidden_bytes // 16) % 2 != 0:
        raise ValueError("DeepEP requires hidden_bytes to be multiple of 32 (bf16 hidden multiple of 16).")


def _run_once(
    args: dict[str, object],
    backend: str | None,
    hidden_states: torch.Tensor,
    token_expert_ids: torch.Tensor,
    token_expert_weights: torch.Tensor,
    ref_out: torch.Tensor | None,
) -> dict[str, object]:
    rank = PM.rank_in("EP")

    dispatcher = TokenDispatcher("EP", num_experts=args["num_experts"])

    if ref_out is not None:
        recv_hidden, _, _ = dispatcher.dispatch(hidden_states, token_expert_ids, token_expert_weights)
        out = dispatcher.combine(recv_hidden)
        torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)

    # Warmup
    for _ in range(args["warmup"]):
        recv_hidden, _, _ = dispatcher.dispatch(hidden_states, token_expert_ids, token_expert_weights)
        _ = dispatcher.combine(recv_hidden)

    dist.barrier()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(args["iters"]):
        recv_hidden, _, _ = dispatcher.dispatch(hidden_states, token_expert_ids, token_expert_weights)
        _ = dispatcher.combine(recv_hidden)
    end.record()

    torch.cuda.synchronize()
    dist.barrier()

    elapsed_ms = start.elapsed_time(end)
    ms_per_iter = elapsed_ms / args["iters"]

    if rank == 0:
        total_tokens = args["seq_len"] * args["ep_size"]
        tokens_per_sec = total_tokens / (ms_per_iter / 1000.0)
        return {
            "ms_per_iter": ms_per_iter,
            "tokens_per_sec": tokens_per_sec,
            "backend": backend or "baseline",
        }
    return {}


def _run_param_set(params: dict[str, object]) -> None:
    did_init = False
    try:
        did_init = _init_dist_and_mesh(ep_size=params["ep_size"])
        target = "steptronoss.model.ep_dispatcher.token_dispatcher.TokenDispatcher"
        if target not in OPTIMIZABLE_REGISTER:
            raise RuntimeError(f"Target not registered: {target}")

        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        hidden_states, token_expert_ids, token_expert_weights = _make_inputs(
            params["seq_len"], params["hidden"], params["topk"], params["num_experts"], device
        )
        ref_out = None
        if params.get("check", False):
            from steptronoss.utils.optimizable import set_optimization

            set_optimization(**{"TokenDispatcher": None})
            ref_dispatcher = TokenDispatcher("EP", num_experts=params["num_experts"])
            recv_hidden, _, _ = ref_dispatcher.dispatch(hidden_states, token_expert_ids, token_expert_weights)
            ref_out = ref_dispatcher.combine(recv_hidden).detach()

        def runner(backend: str | None) -> dict[str, object]:
            if backend == "deep_ep":
                _check_deepep_alignment(params["hidden"])
            return _run_once(
                params,
                backend,
                hidden_states,
                token_expert_ids,
                token_expert_weights,
                ref_out,
            )

        results = run_with_backends(target, runner)
        if PM.rank_in("EP") == 0:
            print(
                f"[dispatcher] ep={params['ep_size']} seq_len={params['seq_len']} "
                f"hidden={params['hidden']} topk={params['topk']}"
            )
            for res in results:
                if not res.ok:
                    print(f"[dispatcher] {res.name}: ERROR {res.error}")
                    continue
                payload = res.payload or {}
                print(
                    f"[dispatcher] {res.name}: {payload.get('ms_per_iter'):.3f} ms/iter, "
                    f"{payload.get('tokens_per_sec'):,.0f} tokens/s (global)"
                )
    finally:
        _cleanup_dist(did_init)


def main() -> int:
    for params in PARAM_SETS:
        _run_param_set(params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
