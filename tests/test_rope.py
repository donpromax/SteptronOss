import pytest
import torch

from steptronoss.model.common.rope import YARNRoPE


def test_yarn_rope_cache_stays_fp32_under_default_bf16():
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        rope = YARNRoPE(dim=8, max_position_embeddings=16)
        assert rope._cos_cache.dtype == torch.float32
        assert rope._sin_cache.dtype == torch.float32

        rope = rope.to(torch.bfloat16)
        assert rope._cos_cache.dtype == torch.float32
        assert rope._sin_cache.dtype == torch.float32

        feature = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
        position_id = torch.arange(4, dtype=torch.int32)
        output = rope(feature, position_id)

        assert output.dtype == torch.bfloat16
        assert rope._cos_cache.dtype == torch.float32
        assert rope._sin_cache.dtype == torch.float32
    finally:
        torch.set_default_dtype(old_default_dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="RoPE cuda cache test requires CUDA")
@pytest.mark.skipif(not torch.cuda.is_bf16_supported(), reason="bf16 not supported")
def test_yarn_rope_cuda_bfloat16_keeps_fp32_cache():
    rope = YARNRoPE(dim=8, max_position_embeddings=16).cuda().bfloat16()

    assert rope._cos_cache.device.type == "cuda"
    assert rope._sin_cache.device.type == "cuda"
    assert rope._cos_cache.dtype == torch.float32
    assert rope._sin_cache.dtype == torch.float32

    feature = torch.randn(1, 4, 1, 8, device="cuda", dtype=torch.bfloat16)
    position_id = torch.arange(4, device="cuda", dtype=torch.int32)
    output = rope(feature, position_id)

    assert output.dtype == torch.bfloat16
    assert rope._cos_cache.device.type == "cuda"
    assert rope._sin_cache.device.type == "cuda"
    assert rope._cos_cache.dtype == torch.float32
    assert rope._sin_cache.dtype == torch.float32
