import os

import torch
from loguru import logger

from steptronoss.utils.patch_utils import StepTronPatchesManager

NPU_PATCH_ACTIVE = False
GROUPED_GEMM_REGISTER_NAME = "steptronoss.model.utils.moe_utils.grouped_gemm"
TOKEN_DISPATCHER_REGISTER_NAME = "steptronoss.model.ep_dispatcher.token_dispatcher.TokenDispatcher"


def _dummy_compile(
    model=None, *, fullgraph=False, dynamic=None, backend="inductor", mode=None, options=None, disable=False, **kwargs
):
    if model is not None:
        return model

    def identity(fn):
        return fn

    return identity


def _torch_npu_available() -> bool:
    try:
        import torch_npu
    except ImportError:
        return False
    return hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available()


def is_npu_active() -> bool:
    return NPU_PATCH_ACTIVE


def get_accel_module():
    return torch.npu if NPU_PATCH_ACTIVE else torch.cuda


def get_runtime_device_type() -> str:
    return "npu" if NPU_PATCH_ACTIVE else "cuda"


def get_lazy_call():
    if NPU_PATCH_ACTIVE and hasattr(torch.npu, "_lazy_call"):
        return torch.npu._lazy_call
    return torch.cuda._lazy_call


def get_accel_device(device=-1):
    device_type = get_runtime_device_type()
    if device == -1:
        return torch.device(device_type)
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, int):
        return torch.device(device_type, device)
    return device


def current_device() -> int:
    return get_accel_module().current_device()


def device_count() -> int:
    return get_accel_module().device_count()


def _set_accel_rng_state(new_state, device=-1):
    module = get_accel_module()
    lazy_call = get_lazy_call()
    accel_device = get_accel_device(device)

    def cb():
        idx = accel_device.index
        if idx is None:
            idx = module.current_device()
        # NOTE: `torch.cuda.default_generators[idx]` raises `IndexError` on Ascend.
        default_generator = module.default_generators[idx]
        default_generator.set_state(new_state)

    lazy_call(cb)


def _clip_grad_norm_fp32_npu(parameters, grads_for_norm, max_norm, norm_type=2, model_parallel_group=None):
    from torch import inf

    from steptronoss.core.utils import multi_tensor_applier, multi_tensor_l2_norm, multi_tensor_scale

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    accel_device = torch.device(get_runtime_device_type(), current_device())
    grads = []
    for param in parameters:
        if param.grad is not None:
            # NOTE: NPU grads are not `torch.cuda.FloatTensor`.
            assert param.grad.dtype == torch.float32, param.grad.dtype
            grads.append(param.grad.detach())

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], device=accel_device, dtype=torch.float)
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group)
        total_norm = total_norm_cuda[0].item()
    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], device=accel_device, dtype=torch.int)
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(multi_tensor_l2_norm, dummy_overflow_buf, [grads_for_norm], False)
            else:
                grad_norm = torch.tensor([0], device=accel_device, dtype=torch.float)
            total_norm = grad_norm**norm_type
        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group)
        total_norm = total_norm.item() ** (1.0 / norm_type)

    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy_overflow_buf = torch.tensor([0], device=accel_device, dtype=torch.int)
        multi_tensor_applier(multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)
    return total_norm


def _count_zeros_fp32_npu(parameters, model_parallel_group):
    from steptronoss.core.tensor_parallel import (
        param_is_not_expert_parallel_duplicate,
        param_is_not_tensor_parallel_duplicate,
    )
    from steptronoss.model.module import param_is_not_shared

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    accel_device = torch.device(get_runtime_device_type(), current_device())
    total_num_zeros = torch.tensor([0], device=accel_device, dtype=torch.float)
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_moe_param = getattr(param, "expert_model_parallel", False)
        if is_moe_param:
            is_not_duplicate = param_is_not_expert_parallel_duplicate(param)
        else:
            is_not_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_duplicate:
            grad = param.grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros
    torch.distributed.all_reduce(total_num_zeros, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group)
    return total_num_zeros.item()


def _fp32_to_float16_npu(val, dtype=torch.bfloat16):
    from torch.autograd import Variable
    from torch.nn.parameter import Parameter

    from steptronoss.model.module import conversion_helper

    def half_conversion(item):
        val_typecheck = item
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = item.data
        # NOTE: NPU fp32 tensors skip the original CUDA-only type check.
        if torch.is_tensor(val_typecheck) and val_typecheck.dtype == torch.float32:
            item = item.to(dtype)
        return item

    return conversion_helper(val, half_conversion)


def _float16_to_fp32_npu(val):
    from torch.autograd import Variable
    from torch.nn.parameter import Parameter

    from steptronoss.model.module import conversion_helper

    def float_conversion(item):
        val_typecheck = item
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = item.data
        # NOTE: NPU fp16/bf16 tensors skip the original CUDA-only type check.
        if torch.is_tensor(val_typecheck) and val_typecheck.dtype in (torch.float16, torch.bfloat16):
            item = item.float()
        return item

    return conversion_helper(val, float_conversion)


def _apply_steptron_ascend_patches():
    logger.info("Applying StepTron Ascend patches.", at=0)
    StepTronPatchesManager.register_patch(
        "steptronoss.core.tensor_parallel.random._set_cuda_rng_state",
        _set_accel_rng_state,
        force_patch=True,
    )
    StepTronPatchesManager.register_patch(
        "steptronoss.core.utils._set_cuda_rng_state",
        _set_accel_rng_state,
        force_patch=True,
    )
    StepTronPatchesManager.register_patch(
        "steptronoss.optimizer.clip_grads.clip_grad_norm_fp32",
        _clip_grad_norm_fp32_npu,
        force_patch=True,
    )
    StepTronPatchesManager.register_patch(
        "steptronoss.optimizer.clip_grads.count_zeros_fp32",
        _count_zeros_fp32_npu,
        force_patch=True,
    )
    StepTronPatchesManager.register_patch(
        "steptronoss.model.module.fp32_to_float16",
        _fp32_to_float16_npu,
        force_patch=True,
    )
    StepTronPatchesManager.register_patch(
        "steptronoss.model.module.float16_to_fp32",
        _float16_to_fp32_npu,
        force_patch=True,
    )
    StepTronPatchesManager.apply_patches()


def apply_npu_patch():
    global NPU_PATCH_ACTIVE
    if not _torch_npu_available():
        raise RuntimeError("NPU patch cannot be applied because torch_npu is not available or NPU is not available.")
    # Patch only once, even if `apply_npu_patch` is called multiple times.
    if not NPU_PATCH_ACTIVE:
        from torch_npu.contrib import transfer_to_npu

        NPU_PATCH_ACTIVE = True
        if os.getenv("STEPTRON_ENABLE_TORCH_COMPILE_ON_NPU", "0") != "1":
            torch.compile = _dummy_compile
        _apply_steptron_ascend_patches()
