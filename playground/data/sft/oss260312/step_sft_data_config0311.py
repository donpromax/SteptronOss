#!/usr/bin/env python3
"""0311 unified recipe plus shared raw-json data config.

This file contains:
- the unified raw-data recipe
- the shared raw-json `Recipe0311DatasetsConfig`
- the shared raw-json `Recipe0311SFTDataConfig`
- the common base used by tokenizer-specific compiled data-config files

How to use:
- large-scale path:
  first compile with a tokenizer-specific file, then use the corresponding
  compiled `SFTDataConfig` in experiments
- direct path:
  import `Recipe0311SFTDataConfig` when you want to train directly from raw json

Notes:
- training directly from raw json is the reference path
- compile is an equivalent acceleration path for large-scale training; it should
  not change dataset semantics
- when compiling, always use the tokenizer path from the actual experiment
"""

from typing import Literal

import torch
from configurize import Ref

from playground.tools.compile_recipe import CompliableDatasetsConfig
from steptronoss.data.recipe import DataRecipe, DataSourceFile
from steptronoss.exp.sft import SFTDataConfig

DATA_ROOT_0311_UNIFIED = "/oss/data/step_sft_data/0311_unified"


GENERAL_FILE_LIST = [
    DataSourceFile(f"{DATA_ROOT_0311_UNIFIED}/general/{basename}")
    for basename in [
        "0311_chunk0000.json",
        "0311_chunk0001.json",
        "0311_chunk0002.json",
        "0311_chunk0003.json",
        "0311_chunk0004.json",
        "0311_chunk0005.json",
        "0311_chunk0006.json",
        "0311_chunk0007.json",
        "0311_chunk0008.json",
        "0311_chunk0009.json",
        "0311_chunk0010.json",
        "0311_chunk0011.json",
        "0311_chunk0012.json",
        "0311_chunk0013.json",
        "0311_chunk0014.json",
        "0311_chunk0015.json",
        "0311_chunk0016.json",
        "0311_chunk0017.json",
        "0311_chunk0018.json",
        "0311_chunk0019.json",
        "0311_chunk0020.json",
        "0311_chunk0021.json",
        "0311_chunk0022.json",
        "0311_chunk0023.json",
        "0311_chunk0024.json",
        "0311_chunk0025.json",
        "0311_chunk0026.json",
        "0311_chunk0027.json",
        "0311_chunk0028.json",
        "0311_chunk0029.json",
        "0311_chunk0030.json",
        "0311_chunk0031.json",
        "0311_chunk0032.json",
        "0311_chunk0033.json",
        "0311_chunk0034.json",
        "0311_chunk0035.json",
        "0311_chunk0036.json",
        "0311_chunk0037.json",
        "0311_chunk0038.json",
        "0311_chunk0039.json",
        "0311_chunk0040.json",
        "0311_chunk0041.json",
        "0311_chunk0042.json",
        "0311_chunk0043.json",
        "0311_chunk0044.json",
        "0311_chunk0045.json",
        "0311_chunk0046.json",
        "0311_chunk0047.json",
        "0311_chunk0048.json",
        "0311_chunk0049.json",
        "0311_chunk0050.json",
        "0311_chunk0051.json",
        "0311_chunk0052.json",
        "0311_chunk0053.json",
        "0311_chunk0054.json",
        "0311_chunk0055.json",
        "0311_chunk0056.json",
        "0311_chunk0057.json",
        "0311_chunk0058.json",
        "0311_chunk0059.json",
        "0311_chunk0060.json",
        "0311_chunk0061.json",
        "0311_chunk0062.json",
        "0311_chunk0063.json",
        "0311_chunk0064.json",
        "0311_chunk0065.json",
        "0311_chunk0066.json",
        "0311_chunk0067.json",
        "0311_chunk0068.json",
        "0311_chunk0069.json",
        "0311_chunk0070.json",
        "0311_chunk0071.json",
        "0311_chunk0072.json",
        "0311_chunk0073.json",
        "0311_chunk0074.json",
        "0311_chunk0075.json",
        "0311_chunk0076.json",
        "0311_chunk0077.json",
        "0311_chunk0078.json",
        "0311_chunk0079.json",
        "0311_chunk0080.json",
        "0311_chunk0081.json",
        "0311_chunk0082.json",
        "0311_chunk0083.json",
        "0311_chunk0084.json",
        "0311_chunk0085.json",
        "0311_chunk0086.json",
        "0311_chunk0087.json",
        "0311_chunk0088.json",
        "0311_chunk0089.json",
        "0311_chunk0090.json",
        "0311_chunk0091.json",
        "0311_chunk0092.json",
        "0311_chunk0093.json",
        "0311_chunk0094.json",
        "0311_chunk0095.json",
        "0311_chunk0096.json",
        "0311_chunk0097.json",
        "0311_chunk0098.json",
        "0311_chunk0099.json",
        "0311_chunk0100.json",
        "0311_chunk0101.json",
        "0311_chunk0102.json",
        "0311_chunk0103.json",
        "0311_chunk0104.json",
        "0311_chunk0105.json",
        "0311_chunk0106.json",
        "0311_chunk0107.json",
        "0311_chunk0108.json",
        "0311_chunk0109.json",
        "0311_chunk0110.json",
        "0311_chunk0111.json",
        "0311_chunk0112.json",
        "0311_chunk0113.json",
        "0311_chunk0114.json",
        "0311_chunk0115.json",
        "0311_chunk0116.json",
        "0311_chunk0117.json",
        "0311_chunk0118.json",
        "0311_chunk0119.json",
        "0311_chunk0120.json",
        "0311_chunk0121.json",
        "0311_chunk0122.json",
        "0311_chunk0123.json",
        "0311_chunk0124.json",
        "0311_chunk0125.json",
        "0311_chunk0126.json",
        "0311_chunk0127.json",
        "0311_chunk0128.json",
        "0311_chunk0129.json",
        "0311_chunk0130.json",
        "0311_chunk0131.json",
        "0311_chunk0132.json",
        "0311_chunk0133.json",
        "0311_chunk0134.json",
        "0311_chunk0135.json",
        "0311_chunk0136.json",
        "0311_chunk0137.json",
        "0311_chunk0138.json",
        "0311_chunk0139.json",
        "0311_chunk0140.json",
        "0311_chunk0141.json",
        "0311_chunk0142.json",
        "0311_chunk0143.json",
        "0311_chunk0144.json",
        "0311_chunk0145.json",
        "0311_chunk0146.json",
        "0311_chunk0147.json",
        "0311_chunk0148.json",
    ]
]


STEP_DATA_RECIPE0311_UNIFIED = DataRecipe(
    domains={
        "general": GENERAL_FILE_LIST,
    },
    epochs={
        "general": 1,
    },
)

SFT_0311_UNIFIED_RECIPE = STEP_DATA_RECIPE0311_UNIFIED


# Datasets Configs
# `Recipe0311DatasetsConfig` is shared by tokenizer variants because raw-json
# loading and template construction are now both based on HF tokenizer paths.
# Use this path directly for debugging, smaller runs, or as the source of
# tokenizer-specific compile flows.
class Recipe0311DatasetsConfig(CompliableDatasetsConfig):
    """Dataset config that reads raw 0311 unified json files directly."""

    max_seq_len: int = 128 * 1024
    """Upper bound used while compiling raw dialogs."""

    tokenizer_path: str = Ref("...tokenizer_cfg.tokenizer_path")
    """Tokenizer path used by compile flow."""

    def get_recipe(self):
        return STEP_DATA_RECIPE0311_UNIFIED

    def get_dataset(self, filelist, template):
        from steptronoss.data.datasets.stepchat_dataset import StepChatJsonDataset

        return StepChatJsonDataset(filelist=filelist, template=template)

    def get_template(self):
        from transformers import AutoTokenizer

        from steptronoss.data.chat_templates.text_template import HuggingFaceTemplate

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return HuggingFaceTemplate(tokenizer=tokenizer)


# Data Config ready for use
# `Recipe0311SFTDataConfig` is the shared raw-json training config.
# Tokenizer-specific compiled configs should inherit from this config so compile
# stays an acceleration-only transformation.
#
# Example:
# class MyRawJsonSFTDataConfig(Recipe0311SFTDataConfig):
#     dataset_cfg = Recipe0311DatasetsConfig
class Recipe0311SFTDataConfig(SFTDataConfig):
    """Ready-to-use SFT data config over raw 0311 unified json files."""

    dataset_cfg = Recipe0311DatasetsConfig

    oversize_policy: Literal["drop", "extend"] = "drop"
    """How to handle samples larger than the target pack length."""

    max_packing_seqlen = Ref("..trainer_cfg.global_seq_length")
    """Target packed sequence length provided by the trainer."""

    seqlen_divisible_by: int = 64
    """Pad packed sequences so lengths align with tensor-parallel needs."""

    global_data_keys = ["cu_seqlens", "position_id"]
    """Batch keys that must be broadcast globally."""

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
        dataloader = async_accelearte_slowfast(dataloader, num_workers=16)
        return dataloader

    def preprocess(self, batch: dict):
        cu_seqlens = batch["cu_seqlens"].to("cuda")
        position_id = batch["position_id"].to("cuda")
        max_seq_len = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        if "tokens" in batch:
            tokens = batch["tokens"].to("cuda")
            labels = batch["labels"].to("cuda")
            loss_masks = batch["loss_mask"].to("cuda")

            return dict(
                input_ids=tokens[None].contiguous(),
                labels=labels[None].contiguous(),
                loss_masks=loss_masks[None].contiguous(),
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
            )
        else:
            return dict(
                cu_seqlens=cu_seqlens,
                max_seq_len=max_seq_len,
                position_id=position_id,
            )

    def pack(self, pieces: list):
        import numpy as np

        size = sum([len(s["tokens"]) - 1 for s in pieces])

        if size % self.seqlen_divisible_by != 0:
            padding_size = self.seqlen_divisible_by - size % self.seqlen_divisible_by
            padding_tensor = np.zeros(padding_size + 1)
            pieces.append({
                "tokens": padding_tensor,
                "loss_mask": padding_tensor,
            })

        sizes = torch.tensor([len(s["tokens"]) - 1 for s in pieces])
        from torch import tensor as T

        tokens = torch.cat([T(s["tokens"][:-1], dtype=torch.long) for s in pieces])
        labels = torch.cat([T(s["tokens"][1:], dtype=torch.long) for s in pieces])
        loss_mask = torch.cat([T(s["loss_mask"][1:], dtype=torch.float32) for s in pieces])

        cu_seqlens = torch.cat([
            torch.zeros(1),
            torch.cumsum(sizes, 0),
        ]).int()

        from steptronoss.utils.general import get_position_id_from_cu_seqlens

        return dict(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            max_seq_len=sizes.max(),
            position_id=get_position_id_from_cu_seqlens(cu_seqlens),
        )
