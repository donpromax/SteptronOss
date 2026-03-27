from steptronoss.utils.npu_patch import apply_npu_patch

apply_npu_patch()  # Ensure the NPU patch is applied before importing any other modules that might use NPU features

import os

from playground.data.sft.oss260312.step_sft_data_config0311_qwen_tokenizer import (
    Recipe0311QwenCompiledSFTDataConfig,
)
from playground.sft.qwen3.qwen3_1p7b_sft_step3_data import Exp as BaseExp
from playground.tools.compile_recipe import (
    CompiledDataRecipe,
    CompiledDatasetsConfig,
)


class AscendRealDatasetsConfig(CompiledDatasetsConfig):
    compiled_recipe = CompiledDataRecipe(
        domains={
            "real_ultrachat": "/oss/steptronoss_data/qwen3_1p7b_sft_real_ultrachat/compiled",
        },
        epochs={
            "real_ultrachat": 1.0,
        },
    )


class AscendRealStep3SFTDataQwenTokenizedConfig(Recipe0311QwenCompiledSFTDataConfig):
    dataset_cfg = AscendRealDatasetsConfig

    def build_dataloader(self, dp_rank=0, dp_size=1):
        from steptronoss.data.dataloader.packed_dataloader import MixedPackedDataloader
        from steptronoss.data.nextable import DPMux, async_accelearte_slowfast

        datasets = self.dataset_cfg.build_datasets()
        dataloader = MixedPackedDataloader(
            datasets=[ds[0] for ds in datasets.values()],
            epochs=[ds[1] for ds in datasets.values()],
            max_length=self.max_packing_seqlen,
            oversize_policy=self.oversize_policy,
            transform=self.pack,
        )
        dataloader = DPMux(dataloader, dp_size=dp_size, dp_rank=dp_rank)

        num_workers = int(os.getenv("STEPTRON_SFT_DATALOADER_WORKERS", "1"))
        return async_accelearte_slowfast(dataloader, num_workers=max(1, num_workers))


class Exp(BaseExp):
    data_cfg = AscendRealStep3SFTDataQwenTokenizedConfig

    def __init__(self):
        super().__init__()
        self.checkpoint_cfg.load_safetensors = "/oss/model_zoo/Qwen3-1.7B-Base/"
        self.checkpoint_cfg.save_dir = "/oss/checkpoints/"


if __name__ == "__main__":
    Exp().train()
