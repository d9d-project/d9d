from typing import Any

import pytest
import torch
from d9d.dataset import (
    DatasetImplementingSampleLengthProtocol,
    SequencePackingDataset,
    StreamingSequencePackingDataset,
    pack_collate,
    pack_samples,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset


def _sample(length: int, offset: int = 0) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(offset, offset + length),
        "position_ids": torch.arange(length),
    }


class _MapDataset(DatasetImplementingSampleLengthProtocol[dict[str, torch.Tensor]], Stateful):
    def __init__(self, lengths: list[int]):
        self._lengths = lengths
        self.state_loaded = False

    def __len__(self) -> int:
        return len(self._lengths)

    def sample_length(self, index: int) -> int:
        return self._lengths[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # input_ids encode the sample index so we can trace which samples landed in a row.
        return _sample(self._lengths[index], offset=index * 1000)

    def state_dict(self) -> dict[str, Any]:
        return {"lengths": self._lengths}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._lengths = state_dict["lengths"]
        self.state_loaded = True


class _StreamDataset(IterableDataset[dict[str, torch.Tensor]], Stateful):
    def __init__(self, lengths: list[int]):
        self._lengths = lengths
        self._cursor = 0

    def __iter__(self):
        while self._cursor < len(self._lengths):
            length = self._lengths[self._cursor]
            sample = _sample(length, offset=self._cursor * 1000)
            self._cursor += 1
            yield sample

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": self._cursor}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cursor = state_dict["cursor"]


@pytest.mark.local
def test_pack_samples_concatenates_and_builds_descriptor() -> None:
    packed = pack_samples([_sample(3), _sample(2), _sample(4)])

    assert packed.tokens["input_ids"].shape == (9,)
    # position_ids restart per segment because each sample carries its own arange.
    assert packed.tokens["position_ids"].tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]
    assert packed.packing.cu_seqlens.tolist() == [0, 3, 5, 9]
    assert packed.packing.cu_seqlens.dtype == torch.int32
    assert packed.packing.max_seqlen == 4


@pytest.mark.local
def test_pack_samples_preserves_nested_structure() -> None:
    # A sample can be any tensor pytree, not just a flat dict; structure is preserved.
    samples = [{"ids": torch.arange(n), "meta": [torch.arange(n), torch.arange(n)]} for n in (3, 2)]
    packed = pack_samples(samples)

    assert packed.tokens["ids"].shape == (5,)
    assert [leaf.shape[0] for leaf in packed.tokens["meta"]] == [5, 5]
    assert packed.packing.cu_seqlens.tolist() == [0, 3, 5]


@pytest.mark.local
def test_pack_samples_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        pack_samples([])


@pytest.mark.local
def test_pack_samples_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        pack_samples([{"a": torch.arange(3), "b": torch.arange(2)}])


@pytest.mark.local
def test_pack_samples_rejects_mismatched_structure() -> None:
    with pytest.raises(ValueError, match="same structure"):
        pack_samples([_sample(2), {"input_ids": torch.arange(2)}])


@pytest.mark.local
def test_packing_dataset_bins_respect_max_length() -> None:
    lengths = [5, 3, 8, 2, 6, 1, 4]
    dataset = SequencePackingDataset.build(_MapDataset(lengths), max_length=10, init_seed=0)

    seen: list[int] = []
    for row_idx in range(len(dataset)):
        row = dataset[row_idx]
        total = row.tokens["input_ids"].shape[0]
        assert total <= 10
        assert row.packing.cu_seqlens[-1].item() == total
        # Recover original sample indices from the encoded offsets.
        seen.extend((row.tokens["input_ids"] // 1000).unique().tolist())

    # Every sample placed exactly once.
    assert sorted(seen) == list(range(len(lengths)))


@pytest.mark.local
def test_packing_dataset_rejects_oversized_sample() -> None:
    with pytest.raises(ValueError, match="exceeding max_length"):
        SequencePackingDataset.build(_MapDataset([4, 20, 3]), max_length=10)


@pytest.mark.local
def test_packing_dataset_rejects_nonpositive_max_length() -> None:
    with pytest.raises(ValueError, match="max_length must be positive"):
        SequencePackingDataset.build(_MapDataset([1, 2]), max_length=0)


@pytest.mark.local
def test_packing_dataset_build_fills_bins_to_exact_token_count() -> None:
    # sample_length is the exact token count, so rows fill to capacity: 4 samples of 3 into rows of 6.
    dataset = SequencePackingDataset.build(_MapDataset([3, 3, 3, 3]), max_length=6, show_progress=False)

    assert [dataset[i].tokens["input_ids"].shape[0] for i in range(len(dataset))] == [6, 6]


@pytest.mark.local
@pytest.mark.parametrize("show_progress", [True, False])
def test_packing_dataset_build_progress_does_not_change_bins(show_progress: bool) -> None:
    lengths = [5, 3, 8, 2, 6, 1, 4]
    dataset = SequencePackingDataset.build(
        _MapDataset(lengths), max_length=10, init_seed=0, show_progress=show_progress
    )

    reference = SequencePackingDataset.build(_MapDataset(lengths), max_length=10, init_seed=0, show_progress=False)

    assert dataset.state_dict()["bins"] == reference.state_dict()["bins"]


@pytest.mark.local
def test_packing_dataset_serves_explicit_bins() -> None:
    dataset = SequencePackingDataset(_MapDataset([2, 3, 1]), bins=[[2, 0], [1]])

    assert len(dataset) == 2
    assert dataset[0].tokens["input_ids"].tolist() == [2000, 0, 1]
    assert dataset[0].packing.cu_seqlens.tolist() == [0, 1, 3]
    assert dataset[1].tokens["input_ids"].tolist() == [1000, 1001, 1002]


@pytest.mark.local
def test_packing_dataset_stateful_roundtrip() -> None:
    lengths = [5, 3, 8, 2, 6, 1, 4, 7, 3]
    original = SequencePackingDataset.build(_MapDataset(lengths), max_length=12, init_seed=7)

    expected = [original[i].tokens["input_ids"].tolist() for i in range(len(original))]

    restored_base = _MapDataset(lengths)
    restored = SequencePackingDataset.build(restored_base, max_length=12, init_seed=999)
    restored.load_state_dict(original.state_dict())

    assert restored_base.state_loaded is True
    assert len(restored) == len(original)
    actual = [restored[i].tokens["input_ids"].tolist() for i in range(len(restored))]
    assert actual == expected


@pytest.mark.local
def test_streaming_dataset_groups_and_flushes_tail() -> None:
    lengths = [4, 4, 4, 3]  # rows: [4,4] (8), [4,3] (7)
    dataset = StreamingSequencePackingDataset(_StreamDataset(lengths), max_length=8)

    rows = list(dataset)

    assert [row.tokens["input_ids"].shape[0] for row in rows] == [8, 7]
    assert rows[0].packing.cu_seqlens.tolist() == [0, 4, 8]
    assert rows[1].packing.cu_seqlens.tolist() == [0, 4, 7]


@pytest.mark.local
def test_streaming_dataset_requires_stateful_base() -> None:
    class _NotStateful(IterableDataset):
        def __iter__(self):
            yield _sample(1)

    with pytest.raises(ValueError, match="Stateful"):
        StreamingSequencePackingDataset(_NotStateful(), max_length=8)


@pytest.mark.local
def test_streaming_dataset_rejects_oversized_sample() -> None:
    dataset = StreamingSequencePackingDataset(_StreamDataset([4, 20]), max_length=8)
    with pytest.raises(ValueError, match="exceeding max_length"):
        list(dataset)


@pytest.mark.local
def test_streaming_dataset_stateful_resume() -> None:
    lengths = [4, 4, 4, 3, 5, 2]
    full = list(StreamingSequencePackingDataset(_StreamDataset(lengths), max_length=8))

    dataset = StreamingSequencePackingDataset(_StreamDataset(lengths), max_length=8)
    iterator = iter(dataset)
    first = next(iterator)
    assert first.tokens["input_ids"].tolist() == full[0].tokens["input_ids"].tolist()

    state = dataset.state_dict()

    resumed = StreamingSequencePackingDataset(_StreamDataset(lengths), max_length=8)
    resumed.load_state_dict(state)
    remaining = [row.tokens["input_ids"].tolist() for row in resumed]

    assert remaining == [row.tokens["input_ids"].tolist() for row in full[1:]]


@pytest.mark.local
def test_pack_collate_adds_batch_dim_and_passes_descriptor() -> None:
    row = pack_samples([_sample(3), _sample(2)])
    collated = pack_collate([row])

    assert collated.tokens["input_ids"].shape == (1, 5)
    assert collated.tokens["position_ids"].shape == (1, 5)
    # Descriptor is passed through unchanged (still 1-D / scalar).
    assert collated.packing.cu_seqlens.tolist() == [0, 3, 5]
    assert collated.packing.max_seqlen == 3


@pytest.mark.local
def test_pack_collate_rejects_multi_row_batch() -> None:
    row = pack_samples([_sample(2)])
    with pytest.raises(ValueError, match="exactly one"):
        pack_collate([row, row])
