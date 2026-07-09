import pytest
import torch
from d9d.core.dist_context import DeviceMeshParameters
from d9d.dataset.batch_iterator import (
    FixedCountMicrobatchPacker,
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
