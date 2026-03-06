# Triton 加速开发流程

本文档总结了在 StepTronOSS 中为模型组件添加 Triton 加速的推荐流程。

目标不只是“写出一个快 kernel”，而是交付一个**正确、可审阅、可 benchmark、可维护**的优化实现。

## 基本原则

- 先看 trace，再决定优化，不靠直觉开工。
- 优化 alternative 必须是**严格等效替换**，不能在调用点按 backend 名字分支。
- 语义接口放在 `steptronoss/model/utils/*`；优化实现放在
  `steptronoss/model/optimizations/*`。
- 融合要选对边界，先做小范围可验证融合，不要默认上来就写大一统 kernel。
- 每个优化都要经过三层验证：单测、microbenchmark、真实实验。

## 1. 先从 Trace 出发

写 Triton 之前，先抓一份真实实验上的短 trace。

重点同时看两层：

1. Python / CPU op
   - `.item()`
   - `.cpu()`
   - `.tolist()`
   - `index_copy_`
2. GPU kernel
   - ATen kernel
   - NCCL kernel
   - 自定义 kernel
   - launch 次数与耗时

先判断瓶颈到底属于：

- host sync
- PyTorch 默认 kernel 不够好
- layout / permutation 开销
- 真正的计算 kernel

如果只是 host sync，优先去掉 host sync；不要默认先写 Triton kernel。

## 2. 选择合适的优化粒度

推荐优先级：

1. 单算子替换
   - 例如：`histogram`、`index_compute`
2. 小范围融合
   - 例如：`index_compute + scatter`
3. 语义级 fused path
   - 例如：`routed_grouped_ffn`

除非 trace 明确证明有必要，否则不要把 routing + grouped GEMM + gather
全部硬塞进一个超大 Triton kernel。

## 3. 优化实现放在 `model/optimizations`

推荐布局：

- 语义接口 / registry：
  - `steptronoss/model/utils/moe_utils.py`
- Triton 实现：
  - `steptronoss/model/optimizations/<component>/triton.py`

单算子推荐写法：

```python
@optimizable(
    alternatives={
        "triton": triton_impl,
    }
)
def semantic_op(...):
    ...
```

融合路径推荐写法：

```python
@optimizable(
    alternatives={
        "fused": triton_fused_impl,
    }
)
def semantic_path(...):
    ...
```

## 4. Alternative 必须是严格等效替换

优化实现必须保持相同的逻辑输入、输出和梯度语义。

规则：

- 不要在调用点根据当前 backend 选择写分支。
- 不要要求调用方准备 backend 专用格式。
- 如果某个 backend 需要特殊处理，把处理藏在 backend 内部。
- 像 `reduce_from_tensor_model_parallel_region(...)` 这种分布式语义边界，除非明确要重构语义，否则不要随便吸进 fused alternative。

坏例子：

```python
if current_backend == "some_backend":
    batch_sizes = batch_sizes.cpu()
```

好例子：

- 语义层只传一个 `batch_sizes`
- 具体 backend 自己做内部归一化

## 5. Import 要独立降级

可选 backend import 必须互相独立。

不要这样：

```python
try:
    import backend_a
    import backend_b
    import backend_c
except:
    backend_a = backend_b = backend_c = None
```

要这样：

- 每个 backend 单独 `try`
- 一个优化 import 失败，不能把别的 backend 一起带崩

## 6. 先做 Forward，再补 Backward

推荐顺序：

1. 先落 forward kernel / fused path
2. 验 forward correctness
3. 再补 backward
4. 再抓 backward trace 看 kernel

分析 backward 时要注意：

- 不要只看 CPU op 名字
- 要重点看 GPU kernel
- checkpoint 会把一部分 forward 重算混进 backward trace，需要区分：
  - 真 backward kernel
  - backward 中的 forward 重算
  - 通信 kernel

## 7. 单测必须马上补

最少需要：

- CPU reference 测试：`tests/test_moe_utils.py`
- GPU forward/backward 等价测试：`tests/test_moe_utils_gpu.py`

推荐覆盖：

- 输出正确性
- 输入梯度正确性
- 参数梯度正确性
- invalid / empty / duplicate route 等边界情况

## 8. 每个优化都必须有 Benchmark

每个 Triton 优化都应该有对应 benchmark，放在 `benchmarks/`。

推荐内容：

- baseline vs optimized alternative
- forward 时间
- backward 时间
- total 时间
- correctness 校验
- 真实模型路径中的代表性 shape

Microbenchmark 必须有，但不能只看 microbenchmark。

## 9. 最后要回到真实实验

做完 microbenchmark 后，要回到真实实验上跑短观测（通常 4-5 iter 就够）。

重点看：

- timer breakdown
- forward/backward trace
- kernel mix
- peak memory
- 收益是否只存在于 warmup 之后

推荐实验入口：

- `playground/sft/step3/step3_toy_sft_step3_data.py`

## 10. 优先优化什么

在 StepTronOSS 里，比较适合 Triton 的目标：

- routing layout
- scatter / gather
- 基于索引的 permutation
- backward 里还在走重型 ATen 索引 kernel 的路径

需要谨慎的目标：

- 已经有强外部 backend 的大 GEMM
- 通信主导的路径
- 分布式语义边界本身

## 11. 实操 Checklist

提交 / handoff 前，至少确认：

- 已抓真实 trace
- 已定位真实瓶颈
- Triton 实现放在 `model/optimizations/`
- 通过 `@optimizable(...)` 注册
- 调用点保持 backend-agnostic
- 已补 CPU / GPU correctness tests
- 已补带 backward + correctness 的 benchmark
- 已跑短真实实验
- 已确认没有 backend-name branching 或隐藏 host sync

## 12. 当前仓库里的参考实现

可以参考这些路径：

- `steptronoss/model/optimizations/moe_routing/triton.py`
- `steptronoss/model/optimizations/moe_gather/triton.py`
- `steptronoss/model/optimizations/moe_scatter/triton.py`
- `steptronoss/model/optimizations/routed_grouped_ffn/triton.py`
