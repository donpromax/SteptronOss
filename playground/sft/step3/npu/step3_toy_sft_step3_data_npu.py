"""
Single-node SFT debugging experiment for the Step3.5 Flash runtime path.

Launch from the repo root with 8 GPUs on one machine:

    torchrun --standalone --nproc-per-node=8 playground/sft/step3/npu/step3_toy_sft_step3_data_npu.py

This experiment is intended to stay aligned with the single-node Step3.5
Flash setup in this repo while swapping in the toy Step3.5 model for faster
debugging. Aside from the toy model definition and checkpoint/weight handling,
the SFT data pipeline and the key runtime settings are kept consistent with
the Step3.5 Flash single-node configuration.
"""

from steptronoss.utils.npu_patch import apply_npu_patch

apply_npu_patch()  # Ensure the NPU patch is applied before importing any other modules that might use NPU features

from playground.sft.step3.step3_toy_sft_step3_data import Exp as BaseExp


class Exp(BaseExp):
    """Toy-model SFT debug config that mirrors the Step3.5 Flash 1-node path."""

    def __init__(self):
        super().__init__()
        self.trainer_cfg.global_seq_length = 8 * 1024

        self.model_cfg.num_layers = 4
        self.model_cfg.swa_layer_list = [False, True, False, True]
        self.model_cfg.parallel_cfg.context_parallel_size = 1
        self.model_cfg.parallel_cfg.tensor_model_parallel_size = 8
        self.model_cfg.tp_cfg.sequence_parallel = True
        self.trainer_cfg.offload_optimizer_state = True

    def configure_optimizable(self):
        from steptronoss.utils.optimizable import set_optimization

        set_optimization(
            TokenDispatcher="npu_alltoall",
            grouped_gemm="npu_gmm",
            AttentionCore="npu-flash-attn",
        )


if __name__ == "__main__":
    Exp().train()
