# Ascend Support

This note documents Ascend/NPU support in StepTronOSS. It focuses on what is
available, how it is enabled, and what runtime constraints matter in practice.

## Overview

StepTronOSS provides three main pieces of Ascend support:

- manual runtime patching through `steptronoss.utils.npu_patch.apply_npu_patch()`
- a native `AttentionCore="npu-flash-attn"` alternative
- native `grouped_gemm="npu_gmm"` and `TokenDispatcher="npu_alltoall"`
  alternatives

The Ascend workflow also includes:

- benchmark scripts for grouped GEMM and token dispatch
- NPU-specific SFT experiment entrypoints
- a small real-data preparation script for Ascend smoke tests
- unit tests for dispatcher layout logic and grouped GEMM correctness

## Tested Environment

The user-facing examples in this document were validated with the following
Ascend software stack:

- CANN `8.3.RC2`
- `torch_npu` `2.8`
- MindSpeed `v2.2.0_core_r0.12.1`

All other dependencies are expected to stay aligned with the project's `uv`
environment.

## Runtime Activation

Ascend runtime patching is enabled manually. Importing `steptronoss` alone does
not activate NPU patches.

To enable the Ascend runtime patch layer, call `apply_npu_patch()` before
importing modules that depend on patched NPU behavior:

```python
from steptronoss.utils.npu_patch import apply_npu_patch

apply_npu_patch()
```

Behavior:

- if `torch_npu` is unavailable, the patch stays inactive and the normal CUDA
  path remains unchanged
- if `torch_npu` is available and `apply_npu_patch()` is called, StepTron:
  - marks the runtime as NPU-backed
  - imports `torch_npu.contrib.transfer_to_npu`
  - optionally replaces `torch.compile` with an identity wrapper
  - patches several CUDA-specific helper paths to use NPU-safe implementations

`apply_npu_patch()` is no longer responsible for registering
`npu-flash-attn`, `npu_gmm`, or `npu_alltoall`. Those alternatives are
declared directly in the corresponding `@optimizable(...)` definitions and
become available when those modules are imported.

## What Gets Patched

When the NPU patch is active, the runtime patches the following:

- CUDA RNG state setters
  Redirected to `torch.npu` generator state handling.
- grad clipping and zero counting helpers
  Adjusted to work on NPU tensors instead of CUDA-only tensor types.
- fp32 <-> fp16/bf16 conversion helpers
  Relaxed to match NPU tensor dtypes.

## Native NPU Alternatives

The NPU optimization backends are registered natively through
`@optimizable(...)`, not through `apply_npu_patch()`:

- `AttentionCore`
  Registers `npu-flash-attn`, implemented by `NpuFlashAttention` in
  `steptronoss/model/common/attention_core.py` and backed by Hugging Face NPU
  flash attention integration.
- `grouped_gemm`
  Registers `npu_gmm`, backed by `mindspeed.ops.gmm.npu_gmm_v2`.
- `TokenDispatcher`
  Registers `npu_alltoall`, backed by
  `steptronoss/model/ep_dispatcher/npu_alltoall_dispatcher.py`.

In practice, most Ascend training and benchmark entrypoints still call
`apply_npu_patch()` early because they rely on the runtime helper patches in
addition to selecting these alternatives.

## `npu_alltoall` Dispatcher

The new dispatcher is a tensorized EP routing backend for NPU.

Key behavior:

- builds a unique token-rank routing map per token, so repeated experts on the
  same remote rank are deduplicated before communication
- uses `all_to_all_single` for hidden states, token ids, and routing weights
- preserves backward through communication with a custom autograd-enabled
  `all_to_all_single`
- on the fast path, tries to use MindSpeed fused
  `npu_moe_token_permute` / `npu_moe_token_unpermute`
- on unsupported cases, falls back to `index_select` / `index_add`-based logic

Fast-path requirements:

- device type must be `npu`
- hidden states must be `torch.bfloat16`
- MindSpeed token permute/unpermute ops must import successfully

The dispatcher exposes two observability flags on the instance:

- `last_dispatch_used_fused_permute`
- `last_combine_used_fused_unpermute`

These are useful in benchmarks to confirm whether the fused path was actually
hit.

## `npu_gmm` Grouped GEMM

The grouped GEMM backend wraps `mindspeed.ops.gmm.npu_gmm_v2`.

Notes:

- the implementation accepts the same semantic inputs as the default
  `grouped_gemm`
- `batch_sizes` is normalized internally to `torch.int64`
- if `batch_sizes` is not already on NPU, it is moved to the input device
- empty input is handled explicitly and returns an empty output with the correct
  shape

Recommended runtime assumptions:

- run in bf16
- keep `batch_sizes` CPU-visible or easy to cast to `torch.int64`

## Environment Variables

The Ascend runtime uses or recognizes these environment variables:

- `STEPTRON_ENABLE_TORCH_COMPILE_ON_NPU=1`
  Keep real `torch.compile` on NPU. Without this, the patch layer replaces
  `torch.compile` with a no-op wrapper to avoid unsupported compile paths.
- `STEPTRON_SFT_DATALOADER_WORKERS`
  Controls async dataloader worker count in the new Qwen3 NPU SFT configs.
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
  Used in the new multi-node Muon SFT resource config.
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
  Also set by that resource config for launch consistency with existing
  training assumptions.

## How To Enable NPU Optimizations

Select the NPU alternatives through the optimization registry. For typical
Ascend runs, you will also call `apply_npu_patch()` early for the runtime patch
layer:

```python
from steptronoss.utils.npu_patch import apply_npu_patch
from steptronoss.utils.optimizable import set_optimization

apply_npu_patch()

set_optimization(
    TokenDispatcher="npu_alltoall",
    grouped_gemm="npu_gmm",
    AttentionCore="npu-flash-attn",
)
```

This pattern is used by the NPU experiment entrypoints below.

## Experiment Entry Points

The following experiment scripts are available:

- `playground/sft/step3/npu/step3_toy_sft_step3_data_npu.py`
  Single-node toy SFT debug path for the Step3.5 Flash runtime.
- `playground/sft/step3/npu/step3p5_flash_sft_step3_data_muon_npu.py`
  Larger Muon-based Step3.5 Flash SFT config for Ascend.
- `playground/sft/qwen3/npu/qwen3_1p7b_sft_step3_data_npu.py`
  Qwen3-1.7B SFT config pointing to a compiled real-data dataset.
- `playground/sft/qwen3/npu/qwen3_30a3b_sft_step3_data_npu.py`
  Qwen3-30A3B SFT config pointing to the same compiled dataset.

Example single-node toy launch:

```bash
torchrun --standalone --nproc-per-node=8 \
  playground/sft/step3/npu/step3_toy_sft_step3_data_npu.py
```

These NPU experiment entrypoints call `apply_npu_patch()` near the top of the
file before importing modules that depend on patched NPU runtime behavior.

The larger Step3.5 Muon config uses a `TorchrunResourceConfig` with
`replica=4` and `gpu=16`, so it should be launched with the repo's usual
multi-node torchrun workflow rather than treated as a one-machine smoke test.

## Real-Data Preparation Script

`playground/sft/qwen3/npu/prepare_ascend_qwen3_real_sft_dataset.py` prepares a
small real SFT dataset for Ascend validation.

It currently:

- streams `HuggingFaceH4/ultrachat_200k` from ModelScope
- normalizes messages into StepTron conversation format
- keeps loss only on assistant turns
- filters invalid, empty, too-short, and too-long samples
- tokenizes with a local Qwen3 tokenizer path
- writes `dialogs.json`, `stats.json`, and compiled dataset shards

Example:

```bash
python playground/sft/qwen3/npu/prepare_ascend_qwen3_real_sft_dataset.py \
  --model-path /oss/model_zoo/Qwen3-1.7B-Base/ \
  --output-root /oss/steptronoss_data/qwen3_1p7b_sft_real_ultrachat
```

This script does not call `apply_npu_patch()`, so dataset preparation does not
depend on NPU runtime activation.

## Tests

The following tests are available:

- `tests/test_npu_alltoall_dispatcher.py`
  Verifies rank bucketing, deduplication, sort contract, and token id dtype
  selection.
- `tests/test_grouped_gemm_npu.py`
  Verifies forward/backward equivalence of `npu_gmm` against a reference grouped
  matmul implementation.

Suggested commands:

```bash
pytest tests/test_npu_alltoall_dispatcher.py
pytest tests/test_grouped_gemm_npu.py
```

The grouped GEMM test is skipped unless an actual NPU runtime is available.

## Benchmarks

The following benchmark scripts are available:

- `benchmarks/benchmark_grouped_gemm_npu.py`
- `benchmarks/benchmark_dispatcher_npu.py`

Example commands:

```bash
python benchmarks/benchmark_grouped_gemm_npu.py
torchrun --nproc_per_node=8 benchmarks/benchmark_dispatcher_npu.py
```

The dispatcher benchmark:

- initializes HCCL distributed state
- builds an EP mesh through `PM`
- compares dispatcher alternatives through `set_optimization`
- measures forward, backward, correctness, and memory

Both NPU benchmark scripts call `apply_npu_patch()` explicitly before using NPU
backends because the benchmarks exercise the runtime patch layer as well as the
registered NPU alternatives.

## Constraints And Fallbacks

Keep these constraints in mind:

- `npu_gmm` requires MindSpeed `npu_gmm_v2`
- the fused dispatcher path requires MindSpeed token permute/unpermute ops
- HCCL is expected for distributed NPU runs
- the fused dispatcher path is currently bf16-only
- if optional MindSpeed ops are unavailable, the dispatcher still works through
  fallback gather/scatter logic, but performance expectations change

## Practical Checklist

Before reporting an Ascend regression, confirm:

- `torch_npu` imports and `torch.npu.is_available()` is true
- if the run depends on patched NPU runtime helpers, `apply_npu_patch()` has
  been called early enough
- `set_optimization(...)` selects `TokenDispatcher="npu_alltoall"` and
  `grouped_gemm="npu_gmm"`, and selects
  `AttentionCore="npu-flash-attn"` when using the NPU flash attention path
- distributed runs use HCCL
- hidden states use bf16 if you expect the fused dispatcher fast path
- MindSpeed optional ops import successfully if you expect peak performance
