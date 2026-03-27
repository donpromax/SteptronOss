# Ascend 支持说明

本文档说明 StepTronOSS 中的 Ascend/NPU 支持，重点覆盖可用能力、启用方式、运行约束，以及对应的实验、测试和 benchmark 入口。

## 概览

StepTronOSS 目前提供三块主要的 Ascend 支持：

- 通过 `steptronoss.utils.npu_patch.apply_npu_patch()` 手动启用运行时补丁
- 原生注册 `AttentionCore="npu-flash-attn"` alternative
- 原生注册 `grouped_gemm="npu_gmm"` 和
  `TokenDispatcher="npu_alltoall"` alternatives

相关工作流还包括：

- grouped GEMM 和 token dispatcher 的 benchmark
- NPU 专用 SFT 实验入口
- 用于 Ascend smoke test 的小规模真实数据准备脚本
- dispatcher 路由逻辑和 grouped GEMM 正确性测试

## 测试环境

本文档中的 Ascend 使用说明基于以下软件栈验证：

- CANN `8.3.RC2`
- `torch_npu` `2.8`
- MindSpeed `v2.2.0_core_r0.12.1`

其余依赖与项目的 `uv` 环境保持一致。

## 运行时启用方式

Ascend 运行时 patch 需要手动启用。仅仅导入 `steptronoss` 不会自动激活
NPU patch。

如果要启用 Ascend 运行时 patch 层，需要在导入依赖补丁后 NPU 行为的模块
之前，显式调用
`apply_npu_patch()`：

```python
from steptronoss.utils.npu_patch import apply_npu_patch

apply_npu_patch()
```

行为如下：

- 如果 `torch_npu` 不可用，补丁不会生效，默认 CUDA 路径保持不变
- 如果 `torch_npu` 可用且调用了 `apply_npu_patch()`，StepTron 会：
  - 标记当前运行时为 NPU
  - 导入 `torch_npu.contrib.transfer_to_npu`
  - 按环境变量决定是否屏蔽 `torch.compile`
  - 替换若干 CUDA-only helper 为 NPU 可用实现

`apply_npu_patch()` 现在不再负责注册 `npu-flash-attn`、`npu_gmm`、
`npu_alltoall`。这些 alternative 直接写在对应模块的
`@optimizable(...)` 定义里，模块导入后即可在 registry 中看到。

## 当前补丁覆盖的内容

当 NPU patch 激活后，运行时会替换这些路径：

- CUDA RNG state setter
  改为使用 `torch.npu` 的随机数状态逻辑。
- grad clipping 和 zero counting helper
  去掉对 CUDA tensor type 的假设，适配 NPU tensor。
- fp32 <-> fp16/bf16 conversion helper
  放宽 dtype 判断，适配 NPU 张量。

## 原生注册的 NPU Alternatives

NPU 优化后端现在通过 `@optimizable(...)` 原生注册，而不是由
`apply_npu_patch()` 动态塞进 registry：

- `AttentionCore`
  注册 `npu-flash-attn`，实现类是
  `steptronoss/model/common/attention_core.py` 里的
  `NpuFlashAttention`，底层走 Hugging Face 的 NPU flash attention 集成。
- `grouped_gemm`
  注册 `npu_gmm`，底层是 `mindspeed.ops.gmm.npu_gmm_v2`。
- `TokenDispatcher`
  注册 `npu_alltoall`，实现位于
  `steptronoss/model/ep_dispatcher/npu_alltoall_dispatcher.py`。

在实际使用里，大多数 Ascend 训练和 benchmark 入口仍然会很早调用
`apply_npu_patch()`，因为除了选择这些 alternative 之外，它们还依赖运行时
helper patch。

## `npu_alltoall` Dispatcher

新的 dispatcher 是一个面向 NPU 的 EP token routing backend。

核心行为：

- 为每个 token 构造唯一的 token-rank 路由映射，同一 token 若 top-k expert 落在同一远端 rank，会先去重再通信
- 用 `all_to_all_single` 传 hidden states、token ids 和 routing weights
- 通过自定义带 autograd 的 `all_to_all_single` 保留反向传播
- 在 fast path 下，尝试使用 MindSpeed 的
  `npu_moe_token_permute` / `npu_moe_token_unpermute`
- 如果不满足 fast path 条件，就回退到 `index_select` / `index_add` 逻辑

fused fast path 的前提：

- device 类型必须是 `npu`
- hidden states 必须是 `torch.bfloat16`
- MindSpeed 的 token permute/unpermute 能正常导入

dispatcher 还暴露了两个观测字段：

- `last_dispatch_used_fused_permute`
- `last_combine_used_fused_unpermute`

benchmark 时可以用它们确认本次是否真的打到了 fused path。

## `npu_gmm` Grouped GEMM

新的 grouped GEMM backend 封装了 `mindspeed.ops.gmm.npu_gmm_v2`。

实现上的注意点：

- 接口保持和默认 `grouped_gemm` 一致
- `batch_sizes` 会在内部归一化为 `torch.int64`
- 如果 `batch_sizes` 不在 NPU 上，会先搬到输入所在设备
- 空输入会显式返回 shape 正确的空输出

推荐的运行假设：

- 以 bf16 为主
- `batch_sizes` 至少要能方便转成 `torch.int64`

## 环境变量

Ascend 运行时会使用或识别这些环境变量：

- `STEPTRON_ENABLE_TORCH_COMPILE_ON_NPU=1`
  保留真实的 `torch.compile`。不设置时，patch 层会把 `torch.compile`
  替换成 no-op wrapper，避开暂不稳定的 compile 路径。
- `STEPTRON_SFT_DATALOADER_WORKERS`
  控制新 Qwen3 NPU SFT 配置中的异步 dataloader worker 数量。
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
  新的 Muon SFT 资源配置里会设置它。
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
  同样由该资源配置设置，用来和现有训练假设保持一致。

## 如何开启 NPU 优化

通过优化 registry 选择 NPU alternative。典型 Ascend 运行里，通常还会在很早
的位置调用 `apply_npu_patch()` 启用运行时 patch 层：

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

这也是下面这些 NPU 实验脚本采用的配置方式。

## 实验入口

可用的实验脚本包括：

- `playground/sft/step3/npu/step3_toy_sft_step3_data_npu.py`
  单机 toy SFT 调试入口，尽量贴近 Step3.5 Flash runtime。
- `playground/sft/step3/npu/step3p5_flash_sft_step3_data_muon_npu.py`
  更完整的 Ascend Step3.5 Flash + Muon SFT 配置。
- `playground/sft/qwen3/npu/qwen3_1p7b_sft_step3_data_npu.py`
  指向编译后真实数据集的 Qwen3-1.7B SFT 配置。
- `playground/sft/qwen3/npu/qwen3_30a3b_sft_step3_data_npu.py`
  指向同一真实数据集的 Qwen3-30A3B SFT 配置。

单机 toy 调试示例：

```bash
torchrun --standalone --nproc-per-node=8 \
  playground/sft/step3/npu/step3_toy_sft_step3_data_npu.py
```

这些 NPU 实验入口都会在文件开头先调用 `apply_npu_patch()`，再导入依赖
补丁后 NPU 运行时行为的模块。

更大的 Step3.5 Muon 配置内部使用了 `TorchrunResourceConfig`，
其中 `replica=4`、`gpu=16`，因此更适合走仓库现有的多机 torchrun
流程，不应把它当成单机 smoke test。

## 真实数据准备脚本

`playground/sft/qwen3/npu/prepare_ascend_qwen3_real_sft_dataset.py`
用于准备 Ascend 验证用的小规模真实 SFT 数据集。

当前脚本会：

- 从 ModelScope 流式读取 `HuggingFaceH4/ultrachat_200k`
- 归一化为 StepTron 对话格式
- 只对 assistant turn 打开 loss
- 过滤非法、空内容、过短、过长样本
- 使用本地 Qwen3 tokenizer 路径做 tokenize
- 输出 `dialogs.json`、`stats.json` 和编译后的 dataset shard

示例：

```bash
python playground/sft/qwen3/npu/prepare_ascend_qwen3_real_sft_dataset.py \
  --model-path /oss/model_zoo/Qwen3-1.7B-Base/ \
  --output-root /oss/steptronoss_data/qwen3_1p7b_sft_real_ultrachat
```

这个脚本不会调用 `apply_npu_patch()`，因此数据准备流程不依赖 NPU
运行时激活。

## 测试

可用的测试包括：

- `tests/test_npu_alltoall_dispatcher.py`
  覆盖 rank bucket、去重、排序契约和 token id dtype 选择逻辑。
- `tests/test_grouped_gemm_npu.py`
  对 `npu_gmm` 做 forward/backward 与 reference grouped matmul 的一致性校验。

推荐命令：

```bash
pytest tests/test_npu_alltoall_dispatcher.py
pytest tests/test_grouped_gemm_npu.py
```

其中 grouped GEMM 测试在没有真实 NPU 运行时的环境下会自动 skip。

## Benchmark

可用的 benchmark 脚本包括：

- `benchmarks/benchmark_grouped_gemm_npu.py`
- `benchmarks/benchmark_dispatcher_npu.py`

示例命令：

```bash
python benchmarks/benchmark_grouped_gemm_npu.py
torchrun --nproc_per_node=8 benchmarks/benchmark_dispatcher_npu.py
```

dispatcher benchmark 会：

- 初始化 HCCL 分布式环境
- 通过 `PM` 建立 EP mesh
- 通过 `set_optimization` 比较不同 dispatcher alternative
- 统计 forward、backward、correctness 和 memory

这两个 NPU benchmark 脚本都会先显式调用 `apply_npu_patch()`，再使用 NPU
backend，因为 benchmark 同时覆盖运行时 patch 层和已注册的 NPU
alternatives。

## 约束与降级路径

需要记住这些运行约束：

- `npu_gmm` 依赖 MindSpeed 的 `npu_gmm_v2`
- dispatcher fused path 依赖 MindSpeed token permute/unpermute
- 分布式 NPU 运行默认假设 backend 是 HCCL
- fused dispatcher 目前只覆盖 bf16 path
- 如果可选的 MindSpeed op 不可用，dispatcher 仍然能走 fallback gather/scatter 逻辑，但性能预期会下降

## 实操检查清单

如果要排查 Ascend 回归，先确认：

- `torch_npu` 能导入，且 `torch.npu.is_available()` 为真
- 如果本次运行依赖 NPU 运行时 helper patch，`apply_npu_patch()` 调用得足够早
- `set_optimization(...)` 已选择 `TokenDispatcher="npu_alltoall"` 和
  `grouped_gemm="npu_gmm"`；如果走 NPU flash attention，还应选择
  `AttentionCore="npu-flash-attn"`
- 分布式运行使用 HCCL
- 如果期望命中 fused dispatcher fast path，hidden states 是 bf16
- 如果期望拿到最高性能，MindSpeed 对应可选 op 可以正常导入
