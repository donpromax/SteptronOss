import types

import pytest
import torch
import torch.distributed as dist

import steptronoss.model.ep_dispatcher.deepep_dispatcher as deepep_dispatcher


class _FakeBuffer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.seen_topk_weights = None

    def get_dispatch_layout(self, token_expert_ids, num_experts):
        del num_experts
        num_tokens = token_expert_ids.shape[0]
        return (
            torch.tensor([num_tokens], device=self.device, dtype=torch.int32),
            torch.tensor([num_tokens], device=self.device, dtype=torch.int32),
            torch.tensor([num_tokens], device=self.device, dtype=torch.int32),
            torch.ones((num_tokens,), device=self.device, dtype=torch.bool),
            None,
        )

    def dispatch(
        self,
        hidden_states,
        handle=None,
        topk_idx=None,
        topk_weights=None,
        num_tokens_per_rank=None,
        num_tokens_per_rdma_rank=None,
        is_token_in_rank=None,
        num_tokens_per_expert=None,
        previous_event=None,
    ):
        del num_tokens_per_rank
        del num_tokens_per_rdma_rank
        del is_token_in_rank
        del num_tokens_per_expert
        del previous_event
        if handle is not None:
            raise AssertionError("cached-handle path is not expected in this test")
        return hidden_states + 1, topk_idx + 1, topk_weights + 2, [], "fake-handle", None

    def combine(self, grad_recv_x, handle=None, topk_weights=None):
        assert handle == "fake-handle"
        self.seen_topk_weights = topk_weights
        grad_hidden = grad_recv_x + 5
        grad_token_probs = None if topk_weights is None else topk_weights + 7
        return grad_hidden, grad_token_probs, None


class _FakeDispatcher:
    def __init__(self, buffer):
        self._buffer = buffer
        self._handle = None
        self._cached_handle = None
        self._cached_recv_token_indices = None
        self._cached_recv_token_probs = None
        self._cached_token_sig = None
        self.num_experts = 4

    def _ensure_buffer(self, hidden_bytes):
        del hidden_bytes
        return self._buffer


@pytest.fixture(scope="session")
def init_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for distributed DeepEP tests")
        return
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported")
        return

    try:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    except Exception as e:
        print(f"Failed to initialize torch.distributed: {e}")
        pytest.skip("Failed to initialize torch.distributed")
        return

    if dist.get_world_size() != 2:
        pytest.skip(
            "Need 2 processes in dist group. "
            "You can run with `torchrun --nproc-per-node=2 -m pytest tests/test_deepep_dispatcher.py`."
        )

    from steptronoss.core.parallel_state import PM

    PM.initialize(backend="nccl")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _set_parallel(tp: int, ep: int):
    from steptronoss.core.parallel_state import PM
    from steptronoss.exp.base_exp import ParallelConfig

    parallel_cfg = ParallelConfig()
    parallel_cfg.tensor_model_parallel_size = tp
    parallel_cfg.pipeline_model_parallel_size = 1
    parallel_cfg.context_parallel_size = 1
    parallel_cfg.expert_model_parallel_size = ep
    parallel_cfg.expert_tensor_parallel_size = tp
    parallel_cfg.virtual_pipeline_model_parallel_size = 1
    PM.set_mesh(parallel_cfg)


@pytest.mark.cpu
def test_deepep_dispatch_backward_propagates_router_weight_grad(monkeypatch):
    monkeypatch.setattr(
        deepep_dispatcher,
        "deep_ep",
        types.SimpleNamespace(topk_idx_t=torch.int32),
    )

    buffer = _FakeBuffer()
    dispatcher = _FakeDispatcher(buffer)

    hidden_states = torch.randn(3, 2, requires_grad=True)
    token_expert_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
    token_expert_weights = torch.randn(3, 2, dtype=torch.float32, requires_grad=True)

    recv_x, recv_token_indices, recv_token_probs = deepep_dispatcher._DeepEPDispatchFn.apply(
        dispatcher,
        hidden_states,
        token_expert_ids,
        token_expert_weights,
    )

    assert recv_token_indices.requires_grad is False
    assert recv_token_probs.requires_grad is True

    loss = recv_x.sum() + recv_token_probs.sum()
    loss.backward()

    assert torch.allclose(hidden_states.grad, torch.full_like(hidden_states, 6))
    assert torch.allclose(token_expert_weights.grad, torch.full_like(token_expert_weights, 8))
    assert torch.allclose(buffer.seen_topk_weights, torch.ones_like(token_expert_weights))


@pytest.mark.xdist_group("torchrun")
@pytest.mark.gpu
@pytest.mark.node2
@pytest.mark.skipif(not torch.cuda.is_available(), reason="deepep dispatcher test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
def test_deepep_dispatcher_preserves_router_grad(monkeypatch, init_dist):
    del init_dist
    from steptronoss.core.parallel_state import PM

    _set_parallel(tp=1, ep=2)

    fake_buffer = _FakeBuffer(device="cuda")
    monkeypatch.setattr(deepep_dispatcher, "HAVE_DEEP_EP", True)
    monkeypatch.setattr(
        deepep_dispatcher,
        "deep_ep",
        types.SimpleNamespace(topk_idx_t=torch.int32),
    )
    monkeypatch.setattr(
        deepep_dispatcher.DeepEPDispatcher,
        "_ensure_buffer",
        lambda self, hidden_bytes: fake_buffer,
    )

    hidden_states = (torch.arange(1, 7, device="cuda", dtype=torch.float32).view(3, 2) + PM.rank_in("EP") * 0.25).to(
        torch.bfloat16
    )
    hidden_states = hidden_states.detach().requires_grad_(True)
    token_expert_ids = torch.tensor([[0, 1], [1, 2], [2, 3]], device="cuda", dtype=torch.int64)
    token_expert_weights = torch.randn(3, 2, device="cuda", dtype=torch.float32).requires_grad_(True)

    dispatcher = deepep_dispatcher.DeepEPDispatcher("EP", num_experts=4)
    recv_hidden, recv_token_indices, recv_token_probs = dispatcher.dispatch(
        hidden_states,
        token_expert_ids,
        token_expert_weights,
    )

    assert recv_token_indices.requires_grad is False
    assert recv_token_probs.requires_grad is True

    loss = recv_hidden.sum() + recv_token_probs.sum()
    loss.backward()

    assert torch.allclose(hidden_states.grad, torch.full_like(hidden_states, 6))
    assert torch.allclose(token_expert_weights.grad, torch.full_like(token_expert_weights, 8))
    assert torch.allclose(fake_buffer.seen_topk_weights, torch.ones_like(token_expert_weights))
