import json

import pytest

from steptronoss.data.datasets.stepchat_dataset import StepChatJsonDataset
from steptronoss.data.recipe import DataSourceFile

pytestmark = pytest.mark.cpu


def _sample_dialog(tag: str) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": f"user-{tag}"},
            {"role": "assistant", "content": f"assistant-{tag}"},
        ],
    }


def test_stepchat_dataset_streaming_and_native_load_match(tmp_path):
    path = tmp_path / "dialogs.json"
    path.write_text(json.dumps([_sample_dialog("0"), _sample_dialog("1"), _sample_dialog("2")]), encoding="utf-8")

    source = DataSourceFile(str(path), subsample_rate=0.67)
    streaming = StepChatJsonDataset._streaming_load_dialog(source)
    native = StepChatJsonDataset._native_load_dialog(source)

    assert streaming[0] == native[0] == str(path)
    assert streaming[2] == native[2]
    assert streaming[1] == native[1]


def test_stepchat_dataset_streaming_and_native_reject_nonstandard_samples(tmp_path):
    path = tmp_path / "dialogs.json"
    path.write_text(
        json.dumps([
            [
                {"role": "user", "content": "user-0"},
                {"role": "assistant", "content": "assistant-0"},
            ]
        ]),
        encoding="utf-8",
    )

    source = DataSourceFile(str(path))

    with pytest.raises(ValueError, match="StepChat sample must be a dict with 'conversations' list"):
        StepChatJsonDataset._streaming_load_dialog(source)

    with pytest.raises(ValueError, match="StepChat sample must be a dict with 'conversations' list"):
        StepChatJsonDataset._native_load_dialog(source)
