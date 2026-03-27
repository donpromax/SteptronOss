import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from modelscope import MsDataset
from torch.utils.data import Dataset

from steptronoss.data.chat_templates.text_template import HuggingFaceTemplate
from steptronoss.data.datasets.compile_dataset import compile_dataset
from steptronoss.tokenizer.hf_compat_tokenizer import HFCompatTokenizer

DEFAULT_DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SPLIT = "train_sft"
DEFAULT_MODEL_PATH = "/oss/model_zoo/Qwen3-1.7B-Base/"
DEFAULT_OUTPUT_ROOT = "/oss/steptronoss_data/qwen3_1p7b_sft_real_ultrachat"


class LocalQwen3Tokenizer(HFCompatTokenizer):
    hf_path = DEFAULT_MODEL_PATH


@dataclass
class PreparedStats:
    raw_seen: int = 0
    kept_dialogs: int = 0
    filtered_missing_messages: int = 0
    filtered_invalid_role: int = 0
    filtered_no_final_assistant: int = 0
    filtered_empty_content: int = 0
    filtered_too_short: int = 0
    filtered_too_long: int = 0


class TokenizedListDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _normalize_messages(sample):
    messages = sample.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None, "missing_messages"

    dialogs = []
    valid_roles = {"system", "user", "assistant"}
    for item in messages:
        role = str(item.get("role", "")).lower().strip()
        if role not in valid_roles:
            return None, "invalid_role"
        content = item.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content if x is not None)
        else:
            content = str(content)
        content = content.strip()
        if not content:
            return None, "empty_content"
        dialogs.append({
            "role": role,
            "content": content,
            "name": "",
            "loss_mask": 1 if role == "assistant" else 0,
            "ground_truth": None,
        })

    if dialogs[-1]["role"] != "assistant":
        return None, "no_final_assistant"

    return {"conversations": dialogs, "images": None}, None


def build_real_dataset(
    dataset_name: str,
    split: str,
    model_path: str,
    output_root: str,
    max_samples: int,
    max_tokens_per_sample: int,
    min_tokens_per_sample: int,
):
    class _Tokenizer(LocalQwen3Tokenizer):
        hf_path = model_path

    tokenizer = _Tokenizer()
    template = HuggingFaceTemplate(tokenizer)

    stats = PreparedStats()
    dialogs = []
    tokenized_items = []

    stream = MsDataset.load(dataset_name, split=split, use_streaming=True)
    for sample in stream:
        stats.raw_seen += 1
        dialog, err = _normalize_messages(sample)
        if err is not None:
            setattr(stats, f"filtered_{err}", getattr(stats, f"filtered_{err}") + 1)
            continue

        tokenized = template(dialog)
        token_count = len(tokenized["tokens"])
        if token_count < min_tokens_per_sample:
            stats.filtered_too_short += 1
            continue
        if token_count > max_tokens_per_sample:
            stats.filtered_too_long += 1
            continue

        dialogs.append(dialog)
        tokenized_items.append(tokenized)
        stats.kept_dialogs += 1
        if stats.kept_dialogs >= max_samples:
            break

    output_root = Path(output_root)
    dialogs_path = output_root / "dialogs.json"
    stats_path = output_root / "stats.json"
    compiled_path = output_root / "compiled"

    os.makedirs(output_root, exist_ok=True)
    with open(dialogs_path, "w", encoding="utf-8") as f:
        json.dump(dialogs, f, ensure_ascii=False)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, ensure_ascii=False, indent=2)

    tokenized_dataset = TokenizedListDataset(tokenized_items)
    compile_dataset(
        tokenized_dataset,
        str(compiled_path),
        sample_meta_extractor=lambda x: len(x["tokens"]),
        num_workers=min(8, os.cpu_count() or 1),
    )

    print(f"Prepared real dataset: {compiled_path}")
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare a small real Qwen3-like SFT dataset for Ascend validation.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--max-tokens-per-sample", type=int, default=1024)
    parser.add_argument("--min-tokens-per-sample", type=int, default=32)
    args = parser.parse_args()

    build_real_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        model_path=args.model_path,
        output_root=args.output_root,
        max_samples=args.max_samples,
        max_tokens_per_sample=args.max_tokens_per_sample,
        min_tokens_per_sample=args.min_tokens_per_sample,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
