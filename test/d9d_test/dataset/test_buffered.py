import random
from typing import Any

import pytest
from d9d.dataset import BufferSortedDataset, DatasetImplementingSortKeyProtocol
from torch.distributed.checkpoint.stateful import Stateful


class MockSortableDataset(DatasetImplementingSortKeyProtocol[int], Stateful):
    def __init__(self, data: list[int]):
        self.data = data
        self.state_loaded = False

    def __len__(self) -> int:
        return len(self.data)

    def sort_key(self, index: int) -> int:
        return self.data[index]

    def __getitem__(self, index: int) -> int:
        return self.data[index]

    def state_dict(self) -> dict[str, Any]:
        return {"data": self.data}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.data = state_dict["data"]
        self.state_loaded = True


@pytest.mark.local
@pytest.mark.parametrize(
    ("dataset_len", "buffer_size", "pack_size"),
    [
        (100, 10, 1),   # Even division, full shuffle
        (100, 10, 5),   # Even division, packs of 5
        (100, 10, 10),  # Pack size == buffer size (no local shuffle, just sorted chunks)
        (100, 30, 1),   # Remainder buffer at end (100 % 30 != 0)
        (100, 30, 5),
        (100, 30, 10),
        (105, 10, 1),   # Odd dataset length
        (105, 10, 5),
        (105, 10, 10),
        (105, 30, 1),   # 105 % 30 = 15 remainder
        (105, 30, 5),
        (105, 30, 10),
    ],
)
def test_buffer_sorted_dataset_coverage_and_len(dataset_len: int, buffer_size: int, pack_size: int):
    raw_data = list(range(dataset_len))
    random.shuffle(raw_data)

    base_dataset = MockSortableDataset(raw_data)
    dataset = BufferSortedDataset(base_dataset, buffer_size=buffer_size, pack_size=pack_size, init_seed=42)

    assert len(dataset) == dataset_len

    output_data = [dataset[i] for i in range(dataset_len)]

    # No data loss or duplication
    assert sorted(output_data) == sorted(raw_data)

    # Length consistency
    assert len(output_data) == dataset_len


@pytest.mark.local
def test_buffer_sorted_dataset_sorting_logic():
    # We want to verify that items are grouped by magnitude, even if shuffled locally
    raw_data = [100, 10, 200, 20]
    base_dataset = MockSortableDataset(raw_data)

    # Buffer=4 covers all data. Pack=2 creates two groups:
    # Sorted Items: [10, 20, 100, 200]
    # Expected Packs: {10, 20} and {100, 200}
    dataset = BufferSortedDataset(base_dataset, buffer_size=4, pack_size=2, init_seed=42)

    outputs = [dataset[i] for i in range(4)]

    # Because of Intra-pack shuffle, we can't compare lists directly.
    # We slice the output into pack-sized chunks and convert to sets.
    chunk_1 = set(outputs[0:2])
    chunk_2 = set(outputs[2:4])

    expected_pack_small = {10, 20}
    expected_pack_large = {100, 200}

    # Because of Inter-pack shuffle, either pack could come first.
    # We verify that we have exactly these two sets.
    assert chunk_1 in (expected_pack_small, expected_pack_large)
    assert chunk_2 in (expected_pack_small, expected_pack_large)
    assert chunk_1 != chunk_2


@pytest.mark.local
def test_buffer_sorted_dataset_insufficient_last_buffer():
    raw_data = [50, 40, 30, 20, 10]
    base_dataset = MockSortableDataset(raw_data)

    dataset = BufferSortedDataset(base_dataset, buffer_size=4, pack_size=2, init_seed=123)

    output = [dataset[i] for i in range(5)]

    # First 4 items come from Buffer 1
    buffer_1_output = output[:4]
    # Last item comes from Buffer 2
    buffer_2_output = output[4:]

    # Check Buffer 1 logic (must be pairs of sorted nums)
    assert sorted(buffer_1_output) == [20, 30, 40, 50]
    # Check pairing strictness
    pair_1 = set(buffer_1_output[:2])
    assert pair_1 in ({20, 30}, {40, 50})

    # Check Buffer 2
    assert buffer_2_output == [10]


@pytest.mark.local
def test_buffer_sorted_dataset_stateful_checkpointing():
    raw_data = list(range(100))
    random.shuffle(raw_data)

    # Setup
    base_dataset = MockSortableDataset(raw_data)
    dataset_orig = BufferSortedDataset(base_dataset, buffer_size=10, pack_size=2, init_seed=999)

    # Parameters
    steps_before_save = 35  # Stop in the middle of a buffer (Buffer 0..9, 10..19, 20..29, 30..39)
    total_len = len(raw_data)

    # Run original dataset halfway
    output_part_1 = []
    for i in range(steps_before_save):
        output_part_1.append(dataset_orig[i])

    # Save
    state_dict = dataset_orig.state_dict()

    # Continue original dataset to get Ground Truth
    output_part_2_expected = []
    for i in range(steps_before_save, total_len):
        output_part_2_expected.append(dataset_orig[i])

    # Restore into new dataset
    base_dataset_new = MockSortableDataset(raw_data)  # Fresh base
    dataset_restored = BufferSortedDataset(base_dataset_new, buffer_size=10, pack_size=2, init_seed=999)
    dataset_restored.load_state_dict(state_dict)

    # Verify sub-component state loading
    assert base_dataset_new.state_loaded is True

    # Run restored dataset
    output_part_2_actual = []
    for i in range(steps_before_save, total_len):
        output_part_2_actual.append(dataset_restored[i])

    assert output_part_2_actual == output_part_2_expected
