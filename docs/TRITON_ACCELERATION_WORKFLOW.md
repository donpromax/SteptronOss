# Triton Acceleration Workflow

This note captures the recommended workflow for adding Triton acceleration to
StepTronOSS model components.

The goal is not just to write a fast kernel, but to land an optimization that is
correct, reviewable, benchmarked, and maintainable.

## Principles

- Optimize from trace evidence, not intuition.
- Prefer strict drop-in alternatives over backend-specific call-site branches.
- Keep semantic APIs in `steptronoss/model/utils/*`; put optimized code in
  `steptronoss/model/optimizations/*`.
- Fuse only where the boundary is stable and measurable.
- Validate at three levels: unit correctness, microbenchmark, real experiment.

## 1. Start From a Trace

Before writing Triton code:

1. Capture a short forward/backward trace on a real experiment.
2. Check both:
   - Python / CPU ops: `.item()`, `.cpu()`, `.tolist()`, `index_copy_`, etc.
   - GPU kernels: ATen kernels, NCCL kernels, custom kernels, launch counts.
3. Decide whether the bottleneck is:
   - host synchronization,
   - poor ATen kernel choice,
   - layout / permutation overhead,
   - or a true compute kernel bottleneck.

Do not start with a giant fused kernel by default.

## 2. Choose the Right Optimization Boundary

Preferred order:

1. Single-op replacement
   - Example: `histogram`, `index_compute`
2. Small local fusion
   - Example: `index_compute + scatter`
3. Semantic fused path
   - Example: `routed_grouped_ffn`

Avoid over-fusing unrelated stages such as routing + grouped GEMM + final gather
into one monolithic Triton kernel unless profiling proves it is necessary.

## 3. Keep Optimized Implementations in `model/optimizations`

Recommended layout:

- Semantic function / registry:
  - `steptronoss/model/utils/moe_utils.py`
- Triton implementation:
  - `steptronoss/model/optimizations/<component>/triton.py`

Example pattern:

```python
@optimizable(
    alternatives={
        "triton": triton_impl,
    }
)
def semantic_op(...):
    ...
```

For fused paths:

```python
@optimizable(
    alternatives={
        "fused": triton_fused_impl,
    }
)
def semantic_path(...):
    ...
```

## 4. Alternatives Must Be Strict Drop-In Replacements

Optimized alternatives should accept the same logical inputs and preserve the
same forward/backward semantics.

Rules:

- Do not branch at call sites based on the currently selected backend.
- Do not require callers to prepare backend-specific argument formats.
- If a backend needs special handling, keep that handling inside the backend.
- Preserve distributed boundaries such as
  `reduce_from_tensor_model_parallel_region(...)` outside fused alternatives
  unless the new implementation intentionally changes that semantic boundary.

Bad pattern:

```python
if current_backend == "some_backend":
    batch_sizes = batch_sizes.cpu()
```

Good pattern:

- Pass one semantic `batch_sizes` input.
- Let the chosen backend normalize internally.

## 5. Be Careful With Imports

Backend imports should be independent.

Bad pattern:

```python
try:
    import backend_a
    import backend_b
    import backend_c
except:
    backend_a = backend_b = backend_c = None
```

Good pattern:

- One `try` block per backend import.
- A failure in one optional optimization must not disable unrelated backends.

## 6. Forward First, Then Backward

Recommended order:

1. Land forward kernel / fused path
2. Validate forward correctness
3. Add backward support
4. Re-profile backward kernels

Backward analysis should focus on GPU kernels, not only CPU op names.

When inspecting backward traces, watch out for checkpoint-induced forward
recompute. Distinguish:

- true backward kernels,
- recomputed forward kernels,
- communication kernels.

## 7. Add Tests Immediately

At minimum:

- CPU reference tests in `tests/test_moe_utils.py`
- GPU forward/backward equivalence tests in `tests/test_moe_utils_gpu.py`

Recommended checks:

- output equality vs baseline
- input gradient equality
- parameter gradient equality
- invalid index / empty / duplicate route edge cases

## 8. Always Add a Benchmark

Every Triton optimization should come with a focused benchmark in `benchmarks/`.

Preferred benchmark contents:

- baseline vs optimized alternatives
- forward time
- backward time
- total time
- correctness verification
- representative sizes from a real model path

Microbenchmarks are necessary, but not sufficient.

## 9. Re-Run a Real Experiment

After microbenchmarking, run a short real experiment (usually 4-5 iterations is
enough) and inspect:

- timer breakdown
- forward/backward traces
- kernel mix
- peak memory
- whether gains survive warmup / compile overhead

Use a real target experiment such as:

- `playground/sft/step3/step3_toy_sft_step3_data.py`

## 10. What to Prioritize

Good Triton targets in StepTronOSS:

- routing layout work
- scatter / gather
- index-based permutation logic
- backward paths still using expensive ATen index kernels

Be cautious when replacing:

- large GEMM paths that already use a strong external backend
- communication-heavy regions
- distributed semantic boundaries

## 11. Practical Checklist

Before merge / handoff:

- Trace the real workload
- Identify the true bottleneck
- Implement the Triton backend in `model/optimizations/`
- Register through `@optimizable(...)`
- Keep semantic call sites backend-agnostic
- Add CPU and GPU correctness tests
- Add benchmark with backward + correctness
- Run a short real experiment and inspect traces
- Verify no hidden host sync or backend-name branching remains

## 12. Current Examples

Useful reference patterns in this repo:

- `steptronoss/model/optimizations/moe_routing/triton.py`
- `steptronoss/model/optimizations/moe_gather/triton.py`
- `steptronoss/model/optimizations/moe_scatter/triton.py`
- `steptronoss/model/optimizations/routed_grouped_ffn/triton.py`
