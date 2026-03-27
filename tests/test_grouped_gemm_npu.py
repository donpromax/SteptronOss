import pytest
import torch

pytestmark = pytest.mark.gpu


GROUPED_GEMM_REGISTER_NAME = "steptronoss.model.utils.moe_utils.grouped_gemm"


def _npu_available() -> bool:
    return hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available()


def _ref_gmm(a: torch.Tensor, b: torch.Tensor, batch_sizes: torch.Tensor, trans_b: bool) -> torch.Tensor:
    batch_sizes_list = batch_sizes.tolist()
    outputs = []
    start = 0
    for i, size in enumerate(batch_sizes_list):
        rhs = b[i].t() if trans_b else b[i]
        outputs.append(a[start : start + size] @ rhs)
        start += size
    if outputs:
        return torch.cat(outputs, dim=0)
    return a.new_zeros((0, b.shape[1] if trans_b else b.shape[2]))


@pytest.mark.skipif(not _npu_available(), reason="grouped_gemm npu_gmm test requires NPU")
@pytest.mark.parametrize("trans_b", [False, True])
def test_npu_gmm_forward_backward(trans_b: bool):
    import steptronoss.utils.npu_patch as npu_patch

    npu_patch.apply_npu_patch()

    from steptronoss.model.utils import grouped_gemm
    from steptronoss.utils.optimizable import OPTIMIZABLE_REGISTER, set_optimization

    if not npu_patch.is_npu_active():
        pytest.skip("StepTron NPU patch is not active")

    register = OPTIMIZABLE_REGISTER[GROUPED_GEMM_REGISTER_NAME]
    assert "npu_gmm" in register["alternatives"]
    print(f"Using npu_gmm optimization: {register['alternatives']['npu_gmm']}")
    assert register["alternatives"]["npu_gmm"].__name__ == "mindspeed_npu_grouped_gemm_v2"

    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_sizes = torch.tensor([3, 1, 4], dtype=torch.int64)
    k = 32
    n = 24

    a = torch.randn(batch_sizes.sum().item(), k, device=device, dtype=torch.bfloat16, requires_grad=True)
    if trans_b:
        b = torch.randn(batch_sizes.numel(), n, k, device=device, dtype=torch.bfloat16, requires_grad=True)
    else:
        b = torch.randn(batch_sizes.numel(), k, n, device=device, dtype=torch.bfloat16, requires_grad=True)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)

    try:
        set_optimization(grouped_gemm="npu_gmm")
        out = grouped_gemm(a, b, batch_sizes, trans_b=trans_b)
        expected = _ref_gmm(a_ref, b_ref, batch_sizes, trans_b)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

        loss = out.float().square().mean()
        expected_loss = expected.float().square().mean()
        loss.backward()
        expected_loss.backward()

        torch.testing.assert_close(a.grad, a_ref.grad, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(b.grad, b_ref.grad, rtol=1e-2, atol=1e-2)
    finally:
        set_optimization(grouped_gemm=None)
