import dataclasses

import pytest
import torch
from d9d.core.dist_context import DeviceMeshParameters
from d9d.dataset import BufferSortedDataset, ShardedDataset, ShardIndexingMode
from d9d.dataset.batch_iterator import (
    FixedCountMicrobatchPacker,
    PinMemoryMicrobatchPackStream,
    num_microbatches_for_global_batch,
)
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader


class SimpleDataset(Dataset):
    def __init__(self, size: int):
        self.data = torch.arange(size, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def simple_collate(batch):
    return torch.stack(batch)


def _make_loader(size, microbatch_size, drop_last=True):
    return StatefulDataLoader(
        SimpleDataset(size),
        batch_size=microbatch_size,
        collate_fn=simple_collate,
        drop_last=drop_last,
    )


@pytest.mark.local
def test_num_microbatches_for_global_batch_non_distributed(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    assert num_microbatches_for_global_batch(dist_ctx, global_batch_size=32, microbatch_size=8) == 4
    assert num_microbatches_for_global_batch(dist_ctx, global_batch_size=8, microbatch_size=8) == 1


@pytest.mark.distributed
def test_num_microbatches_for_global_batch_distributed(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))

    assert num_microbatches_for_global_batch(dist_ctx, global_batch_size=128, microbatch_size=8) == 2
    assert num_microbatches_for_global_batch(dist_ctx, global_batch_size=64, microbatch_size=8) == 1


@pytest.mark.distributed
def test_num_microbatches_for_global_batch_indivisible_distributed(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))

    with pytest.raises(ValueError, match="divisible"):  # 8 * 8 = 64 ; 32 < 64
        num_microbatches_for_global_batch(dist_ctx, global_batch_size=32, microbatch_size=8)
    with pytest.raises(ValueError, match="divisible"):
        num_microbatches_for_global_batch(dist_ctx, global_batch_size=65, microbatch_size=8)


@pytest.mark.local
@pytest.mark.parametrize(
    ("size", "microbatch_size", "microbatches_per_step", "drop_last", "expected"),
    [
        # Exact fit: 6 microbatches grouped 3-per-pack -> 2 full packs (drop_last irrelevant).
        (
            12,
            2,
            3,
            True,
            [
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
            ],
        ),
        # Remainder, drop_last=True: 5 microbatches grouped 2-per-pack -> 2 full packs, trailing one dropped.
        (
            10,
            2,
            2,
            True,
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
            ],
        ),
        # Remainder, drop_last=False: same stream -> 2 full packs + 1 short trailing pack kept.
        (
            10,
            2,
            2,
            False,
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
                [[8.0, 9.0]],
            ],
        ),
    ],
)
def test_fixed_count_packer_groups_into_packs(size, microbatch_size, microbatches_per_step, drop_last, expected):
    loader = _make_loader(size=size, microbatch_size=microbatch_size)
    packer = FixedCountMicrobatchPacker(loader, microbatches_per_step=microbatches_per_step, drop_last=drop_last)

    assert packer.total_steps == len(expected)

    packs = list(packer)

    # Every microbatch appears once, in order, grouped correctly.
    contents = [[mb.tolist() for mb in pack] for pack in packs]
    assert contents == expected

    # microbatches stay on CPU - moving to device is the loop's job
    for pack in packs:
        for microbatch in pack:
            assert isinstance(microbatch, torch.Tensor)
            assert microbatch.device == torch.device("cpu")


@pytest.mark.local
def test_fixed_count_packer_requires_positive_count():
    loader = _make_loader(size=10, microbatch_size=2)

    with pytest.raises(ValueError, match="positive"):
        FixedCountMicrobatchPacker(loader, microbatches_per_step=0)


@pytest.mark.local
def test_fixed_count_packer_state_delegates_to_loader():
    loader = _make_loader(size=12, microbatch_size=2)
    packer = FixedCountMicrobatchPacker(loader, microbatches_per_step=2)

    it = iter(packer)
    next(it)  # consumes 2 microbatches
    state = packer.state_dict()

    new_packer = FixedCountMicrobatchPacker(_make_loader(size=12, microbatch_size=2), microbatches_per_step=2)
    new_packer.load_state_dict(state)

    resumed_pack = next(iter(new_packer))
    assert resumed_pack[0].tolist() == [4.0, 5.0]


@dataclasses.dataclass
class _Microbatch:
    ids: torch.Tensor
    meta: dict[str, torch.Tensor]
    size: int


class _PackStream:
    total_steps = 2

    def __iter__(self):
        for step in range(self.total_steps):
            ids = torch.arange(step * 4, step * 4 + 4)
            yield [_Microbatch(ids=ids, meta={"double": ids * 2}, size=4)]

    def state_dict(self):
        return {"marker": 7}

    def load_state_dict(self, state_dict):
        self.loaded = state_dict


@pytest.mark.local
def test_pin_memory_stream_pins_tensors_nested_in_dataclasses():
    packs = list(PinMemoryMicrobatchPackStream(_PackStream()))

    assert len(packs) == 2
    for step, pack in enumerate(packs):
        (microbatch,) = pack
        assert isinstance(microbatch, _Microbatch)
        assert microbatch.ids.is_pinned()
        assert microbatch.meta["double"].is_pinned()
        assert microbatch.ids.tolist() == list(range(step * 4, step * 4 + 4))
        assert microbatch.size == 4


@pytest.mark.local
def test_pin_memory_stream_delegates_total_steps_and_state():
    inner = _PackStream()
    stream = PinMemoryMicrobatchPackStream(inner)

    assert stream.total_steps == 2
    assert stream.state_dict() == {"marker": 7}

    stream.load_state_dict({"marker": 8})
    assert inner.loaded == {"marker": 8}


class _SortableDataset(SimpleDataset):
    def sort_key(self, idx):
        return int(idx) % 7


def _freeze(state):
    if isinstance(state, torch.Tensor):
        return ("tensor", tuple(state.flatten().tolist()))
    if isinstance(state, dict):
        return ("dict", tuple(sorted((repr(key), _freeze(value)) for key, value in state.items())))
    if isinstance(state, (list, tuple)):
        return ("seq", tuple(_freeze(value) for value in state))
    return ("value", repr(state))


@pytest.mark.local
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize(
    "make_dataset",
    [
        pytest.param(lambda: SimpleDataset(64), id="plain"),
        pytest.param(
            lambda: BufferSortedDataset(_SortableDataset(64), buffer_size=16, pack_size=4, init_seed=0),
            id="buffer_sorted",
        ),
        pytest.param(
            lambda: ShardedDataset(
                SimpleDataset(64),
                total_shards=2,
                current_shard=1,
                indexing_mode=ShardIndexingMode.sequential,
                pad_to_equal_size_across_shards=True,
            ),
            id="sharded",
        ),
    ],
)
def test_builtin_stream_state_snapshot_is_not_mutated_by_further_iteration(num_workers, make_dataset):
    # The MicrobatchPackStream contract prefetching relies on: a state snapshot stays valid while the stream
    # keeps being iterated.
    loader = StatefulDataLoader(
        make_dataset(), batch_size=2, collate_fn=simple_collate, num_workers=num_workers, shuffle=True
    )
    stream = FixedCountMicrobatchPacker(loader, microbatches_per_step=2)
    iterator = iter(stream)
    next(iterator)

    snapshot = stream.state_dict()
    frozen = _freeze(snapshot)
    for _ in range(3):
        next(iterator)

    assert _freeze(snapshot) == frozen
