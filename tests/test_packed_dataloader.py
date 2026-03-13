import pytest
from torch.utils.data import Dataset

from steptronoss.data.dataloader.packed_dataloader import MixedPackedDataloader
from steptronoss.data.samplers.base_sampler import LoopedShuffleSampler

pytestmark = pytest.mark.cpu


class SizedDataset(Dataset):
    def __init__(self, sizes: list[int], tag: int):
        self._sizes = sizes
        self._tag = tag

    def __len__(self) -> int:
        return len(self._sizes)

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self._sizes)
        if idx < 0 or idx >= len(self._sizes):
            raise IndexError("index out of range")
        return [self._tag] * self._sizes[idx]


def _build_loader():
    ds_a = SizedDataset([2, 3], tag=0)
    ds_b = SizedDataset([1, 4], tag=1)
    return MixedPackedDataloader(
        datasets=[ds_a, ds_b],
        epochs=[1.0, 1.0],
        max_length=5,
        oversize_policy="extend",
    )


def test_packed_dataloader_get_and_len():
    loader = _build_loader()

    expected = []
    for start, count in loader.packing_result.packed_sample_ranges:
        items = []
        total_len = 0
        for domain_idx, in_domain_idx in loader.piece_order[start : start + count]:
            item = loader.datasets[domain_idx][in_domain_idx]
            items.append(item)
            total_len += len(item)
        expected.append(items)
        assert total_len <= 5

    assert len(loader) == len(expected)

    for items in expected:
        assert loader.get() == items
        loader.update()

    assert loader.get() == expected[0]


def test_packed_dataloader_state_roundtrip():
    loader = _build_loader()
    for _ in range(3):
        next(loader)

    state = loader.state_dict()
    loader2 = _build_loader()
    loader2.load_state_dict(state)

    seq1 = [next(loader) for _ in range(6)]
    seq2 = [next(loader2) for _ in range(6)]
    assert seq1 == seq2


def test_packed_dataloader_supports_per_dataset_sampling():
    ds_a = SizedDataset([1, 1, 1], tag=0)
    ds_b = SizedDataset([1, 1, 1], tag=1)
    loader = MixedPackedDataloader(
        datasets=[ds_a, ds_b],
        epochs=[1.0, 1.0],
        max_length=1,
        oversize_policy="extend",
        dataset_sampling=["sequential", "random"],
    )

    domain_a_indices = [in_domain_idx for domain_idx, in_domain_idx in loader.piece_order if domain_idx == 0]
    domain_b_indices = [in_domain_idx for domain_idx, in_domain_idx in loader.piece_order if domain_idx == 1]

    assert domain_a_indices == [0, 1, 2]

    sampler_b = LoopedShuffleSampler(size=3, base_seed=1235)
    expected_domain_b_indices = [next(sampler_b) for _ in range(3)]
    assert domain_b_indices == expected_domain_b_indices


def test_packed_dataloader_rejects_mismatched_dataset_sampling_length():
    ds_a = SizedDataset([1], tag=0)
    ds_b = SizedDataset([1], tag=1)

    with pytest.raises(ValueError, match="dataset_sampling list must have the same length as datasets"):
        MixedPackedDataloader(
            datasets=[ds_a, ds_b],
            epochs=[1.0, 1.0],
            max_length=1,
            dataset_sampling=["random"],
        )
