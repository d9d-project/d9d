import pytest
import torch
from d9d.core.dist_context import DeviceMeshParameters
from d9d.dataset import FixedCountMicrobatchPacker, ShardedDataset, ShardIndexingMode
from d9d.loop.auto import AutoDataConfig, AutoDataProvider
from d9d.loop.control import InitializeDataProviderContext
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


@pytest.fixture
def context(dist_ctx_factory):
    return InitializeDataProviderContext(dist_context=dist_ctx_factory(DeviceMeshParameters()))


@pytest.mark.local
def test_auto_data_provider_passes_dist_context_to_factory(context):
    seen = []

    def dataset_factory(dist_context):
        seen.append(dist_context)
        return SimpleDataset(12)

    config = AutoDataConfig(global_batch_size=6, microbatch_size=2, num_workers=0, pin_memory=False)
    provider = AutoDataProvider(dataset_factory=dataset_factory, collator=simple_collate, config=config)

    provider(context)

    assert seen == [context.dist_context]


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

    loader = provider(context)._loader

    assert isinstance(loader, StatefulDataLoader)
    assert loader.batch_size == config.microbatch_size
    assert loader.collate_fn is simple_collate
    assert loader.num_workers == config.num_workers
    assert loader.pin_memory == config.pin_memory
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
