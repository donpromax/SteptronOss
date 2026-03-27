from steptronoss.utils.npu_patch import apply_npu_patch

apply_npu_patch()  # Ensure the NPU patch is applied before importing any other modules that might use NPU features


from playground.sft.step3.step3p5_flash_sft_step3_data_muon import Exp as BaseExp
from steptronoss.exp.resources import TorchrunResourceConfig


class Step3F128kSFTResourceConfig(TorchrunResourceConfig):
    def __init__(self):
        super().__init__()
        self.replica = 4
        self.gpu = 16
        self.envs |= {
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }


class Exp(BaseExp):
    log_dir = "/data/logs/"

    resource_cfg = Step3F128kSFTResourceConfig

    def __init__(self):
        super().__init__()
        self.trainer_cfg.global_seq_length = 1024 * 8

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
