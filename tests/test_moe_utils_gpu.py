import functools

import pytest
import torch
import torch.nn.functional as F

from steptronoss.model.optimizations.moe_routing.triton import triton_index_scatter
from steptronoss.model.utils.moe_utils import histogram, index_compute, routed_grouped_ffn
from steptronoss.utils.optimizable import set_optimization

pytestmark = pytest.mark.gpu


def _make_topk_ids(
    token_num: int,
    topk: int,
    num_experts: int,
    device: torch.device,
    *,
    allow_dupe: bool = False,
    invalid_frac: float = 0.0,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="histogram triton test requires CUDA")
@pytest.mark.parametrize(
    "dtype,allow_dupe,invalid_frac",
    [
        (torch.int32, False, 0.0),
        (torch.int64, False, 0.3),
        (torch.int64, True, 0.1),
    ],
)
def test_histogram_triton_matches_reference(dtype, allow_dupe, invalid_frac):
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk_ids = _make_topk_ids(
        token_num=4096,
        topk=2,
        num_experts=64,
        device=device,
        allow_dupe=allow_dupe,
        invalid_frac=invalid_frac,
        dtype=dtype,
    )

    try:
        set_optimization(histogram=None)
        ref = histogram(topk_ids, expert_num=64)
        set_optimization(histogram="triton")
        out = histogram(topk_ids, expert_num=64)
        torch.testing.assert_close(out, ref)
    finally:
        set_optimization(histogram=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="index_compute triton test requires CUDA")
@pytest.mark.parametrize(
    "dtype,allow_dupe,invalid_frac",
    [
        (torch.int32, False, 0.0),
        (torch.int64, False, 0.2),
        (torch.int64, True, 0.0),
    ],
)
def test_index_compute_triton_matches_reference(dtype, allow_dupe, invalid_frac):
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk_ids = _make_topk_ids(
        token_num=4096,
        topk=2,
        num_experts=64,
        device=device,
        allow_dupe=allow_dupe,
        invalid_frac=invalid_frac,
        dtype=dtype,
    )

    try:
        set_optimization(histogram=None, index_compute=None)
        experts_hist = histogram(topk_ids, expert_num=64)
        ref = index_compute(topk_ids, experts_hist)
        set_optimization(index_compute="triton")
        out = index_compute(topk_ids, experts_hist)
        torch.testing.assert_close(out, ref)
    finally:
        set_optimization(histogram=None, index_compute=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="index_scatter triton test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
def test_index_scatter_triton_matches_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    token_num = 2048
    hidden = 256
    topk = 2
    num_experts = 64

    x = torch.randn((token_num, hidden), device=device, dtype=torch.bfloat16, requires_grad=True)
    topk_ids = _make_topk_ids(
        token_num=token_num,
        topk=topk,
        num_experts=num_experts,
        device=device,
        allow_dupe=False,
        invalid_frac=0.2,
        dtype=torch.int64,
    )

    experts_hist = histogram(topk_ids, expert_num=num_experts)
    scatter_ref = index_compute(topk_ids, experts_hist)

    from steptronoss.model.utils.moe_utils import moe_scatter

    x_ref = x.detach().clone().requires_grad_(True)
    out_ref = moe_scatter(x_ref, scatter_ref)
    out_ref.sum().backward()

    x_run = x.detach().clone().requires_grad_(True)
    out, scatter = triton_index_scatter(x_run, topk_ids, experts_hist)
    out.sum().backward()

    torch.testing.assert_close(scatter, scatter_ref)
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x_run.grad, x_ref.grad, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="routed_grouped_ffn fused test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
def test_routed_grouped_ffn_fused_matches_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    token_num = 256
    hidden = 64
    topk = 2
    num_experts = 8
    ffn_hidden = 96

    x = torch.randn((token_num, hidden), device=device, dtype=torch.bfloat16, requires_grad=True)
    token_expert_ids = _make_topk_ids(
        token_num=token_num,
        topk=topk,
        num_experts=num_experts,
        device=device,
        allow_dupe=False,
        invalid_frac=0.15,
        dtype=torch.int64,
    )
    token_weights = torch.rand((token_num, topk), device=device, dtype=torch.float32)
    token_weights = token_weights / token_weights.sum(dim=1, keepdim=True)
    w1 = torch.randn((num_experts, 2 * ffn_hidden, hidden), device=device, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn((num_experts, hidden, ffn_hidden), device=device, dtype=torch.bfloat16, requires_grad=True)
    act = functools.partial(
        lambda x, swiglu_limit=None: F.silu(torch.chunk(x, 2, dim=-1)[0]) * torch.chunk(x, 2, dim=-1)[1]
    )

    try:
        set_optimization(
            histogram=None,
            index_compute=None,
            moe_scatter=None,
            moe_weighted_gather=None,
            grouped_gemm=None,
            routed_grouped_ffn=None,
        )
        x_ref = x.detach().clone().requires_grad_(True)
        w1_ref = w1.detach().clone().requires_grad_(True)
        w2_ref = w2.detach().clone().requires_grad_(True)
        out_ref = routed_grouped_ffn(
            w1_ref,
            w2_ref,
            act,
            x_ref,
            token_expert_ids,
            token_weights,
        )
        out_ref.sum().backward()

        set_optimization(
            histogram=None,
            index_compute=None,
            moe_scatter=None,
            moe_weighted_gather=None,
            grouped_gemm=None,
            routed_grouped_ffn="fused",
        )
        out = routed_grouped_ffn(
            w1,
            w2,
            act,
            x,
            token_expert_ids,
            token_weights,
        )
        out.sum().backward()

        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(w1.grad, w1_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(w2.grad, w2_ref.grad, rtol=1e-2, atol=1e-2)
    finally:
        set_optimization(
            histogram=None,
            index_compute=None,
            moe_scatter=None,
            moe_weighted_gather=None,
            grouped_gemm=None,
            routed_grouped_ffn=None,
        )
