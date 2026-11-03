import pytest
import torch
from d9d.data import (
    FixedCountMicrobatchPacker,
    PinMemoryMicrobatchPackStream,
    ShardedDataset,
    ShardIndexingMode,
)
from d9d.dist_context import DeviceMeshParameters
from d9d.loop.auto import AutoDataConfig, AutoDataProvider
from d9d.loop.control import InitializeDataProviderContext
from d9d.rng import SeedScope
from torch.utils.data import Dataset, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler


class SimpleDataset(Dataset):
    def __init__(self, size: int):
        self.data = torch.arange(size, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def simple_collate(batch):
    return torch.stack(batch)


def _build_context(dist_context, shard_seed_scope):
    return InitializeDataProviderContext(
        dist_context=dist_context, seed_scope=SeedScope.root(0).child("data"), shard_seed_scope=shard_seed_scope
    )


@pytest.fixture
def context(dist_ctx_factory):
    return _build_context(dist_ctx_factory(DeviceMeshParameters()), SeedScope.root(0).child("data", "shard", 0))


@pytest.mark.local
def test_auto_data_provider_passes_context_to_factory(context):
    seen = []

    def dataset_factory(factory_context):
        seen.append(factory_context)
        return SimpleDataset(12)

    config = AutoDataConfig(global_batch_size=6, microbatch_size=2, num_workers=0, pin_memory=False)
    provider = AutoDataProvider(dataset_factory=dataset_factory, collator=simple_collate, config=config)

    provider(context)

    assert seen == [context]


@pytest.mark.local
def test_auto_data_provider_shuffles_by_shard_seed_scope(dist_ctx_factory):
    dist_context = dist_ctx_factory(DeviceMeshParameters())
    config = AutoDataConfig(global_batch_size=32, microbatch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    provider = AutoDataProvider(dataset_factory=lambda _ctx: SimpleDataset(32), collator=simple_collate, config=config)

    def read_order(shard_seed_scope, global_seed):
        # The loader must not depend on the global generator.
        torch.manual_seed(global_seed)
        (pack,) = provider(_build_context(dist_context, shard_seed_scope))
        return pack[0].tolist()

    scope = SeedScope.root(0).child("data", "shard", 0)
    other_scope = SeedScope.root(0).child("data", "shard", 1)

    assert read_order(scope, global_seed=1) == read_order(scope, global_seed=2)
    assert read_order(scope, global_seed=1) != read_order(other_scope, global_seed=1)


@pytest.mark.local
@pytest.mark.parametrize(
    "config",
    [
        AutoDataConfig(global_batch_size=6, microbatch_size=2, num_workers=0, pin_memory=False),
        AutoDataConfig(
            global_batch_size=8,
            microbatch_size=4,
            shuffle=True,
            drop_last=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            timeout=7.0,
        ),
        AutoDataConfig(
            global_batch_size=12,
            microbatch_size=3,
            shuffle=False,
            drop_last=True,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            timeout=0.5,
        ),
    ],
)
def test_auto_data_provider_wires_config_into_loader(context, config):
    # The provider maps AutoDataConfig fields onto the underlying StatefulDataLoader kwargs
    # (microbatch_size becomes batch_size). We assert the loader received the configured values,
    # not that torch acts on them.
    provider = AutoDataProvider(dataset_factory=lambda _ctx: SimpleDataset(12), collator=simple_collate, config=config)

    stream = provider(context)
    # pinning is done by the stream wrapper, which traverses dataclasses, not by the loader
    assert isinstance(stream, PinMemoryMicrobatchPackStream) == config.pin_memory
    packer = stream._inner if isinstance(stream, PinMemoryMicrobatchPackStream) else stream
    loader = packer._loader

    assert isinstance(loader, StatefulDataLoader)
    assert loader.batch_size == config.microbatch_size
    assert loader.collate_fn is simple_collate
    assert loader.num_workers == config.num_workers
    assert loader.pin_memory is False
    assert loader.persistent_workers == config.persistent_workers
    assert loader.prefetch_factor == config.prefetch_factor
    assert loader.timeout == config.timeout
    assert loader.drop_last == config.drop_last
    expected_sampler = RandomSampler if config.shuffle else SequentialSampler
    assert isinstance(loader.sampler, expected_sampler)


@pytest.mark.local
def test_auto_data_provider_shards_dataset_with_configured_mode(context):
    config = AutoDataConfig(
        global_batch_size=6,
        microbatch_size=2,
        shard_indexing_mode=ShardIndexingMode.chunked,
        num_workers=0,
        pin_memory=False,
    )
    provider = AutoDataProvider(dataset_factory=lambda _ctx: SimpleDataset(12), collator=simple_collate, config=config)

    dataset = provider(context)._loader.dataset

    # The provider wraps the dataset for data-parallel sharding, forwarding the indexing mode.
    assert isinstance(dataset, ShardedDataset)
    assert dataset._indexing_mode == ShardIndexingMode.chunked


@pytest.mark.local
def test_auto_data_provider_wires_pack_size_and_drop_last_into_packer(context):
    dropping = AutoDataProvider(
        dataset_factory=lambda _ctx: SimpleDataset(10),
        collator=simple_collate,
        config=AutoDataConfig(global_batch_size=4, microbatch_size=2, drop_last=True, num_workers=0, pin_memory=False),
    )
    keeping = AutoDataProvider(
        dataset_factory=lambda _ctx: SimpleDataset(10),
        collator=simple_collate,
        config=AutoDataConfig(global_batch_size=4, microbatch_size=2, drop_last=False, num_workers=0, pin_memory=False),
    )

    dropping_stream = dropping(context)
    keeping_stream = keeping(context)

    # The default stack groups microbatches with a FixedCountMicrobatchPacker.
    assert isinstance(dropping_stream, FixedCountMicrobatchPacker)
    assert isinstance(keeping_stream, FixedCountMicrobatchPacker)

    assert [len(pack) for pack in dropping_stream] == [2, 2]
    assert [len(pack) for pack in keeping_stream] == [2, 2, 1]
