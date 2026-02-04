import dataclasses

import pytest
import torch
from d9d.loop.component.data_loader_factory import (
    DataLoaderFactory,
    IteratorBatchGroup,
    StatefulDataLoaderDataParallelAware,
)
from d9d.loop.config import DataLoadingConfig
from d9d.loop.control import InitializeDatasetContext, InitializeDatasetResult
from torch.testing import assert_close
from torch.utils.data import Dataset


@dataclasses.dataclass
class MockBatchMaths:
    data_loader_batch_size: int = 4
    num_microbatches_gradient_accumulation: int = 2


class SimpleDataset(Dataset):
    def __init__(self, size: int):
        self.data = torch.arange(size, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def simple_collate(batch):
    return torch.stack(batch)


@pytest.mark.parametrize(
    ("num_items", "batch_group_size", "expected_groups"),
    [
        (6, 2, [2, 2, 2]),  # Exact fit
        (5, 2, [2, 2, 1]),  # Remainder
        (1, 1, [1]),  # Single item
        (2, 3, [2]),  # Less than group size
        (0, 3, []),  # Empty
    ]
)
@pytest.mark.local
def test_iterator_batch_group(num_items, batch_group_size, expected_groups):
    data = [torch.tensor(i, device="cpu") for i in range(num_items)]
    base_iter = iter(data)
    device = torch.tensor(0).to("cuda").device

    grouper = IteratorBatchGroup(base_iter, device, batch_group_size)

    group_sizes = []

    total_counter = 0
    for group_iter in grouper:
        group_size = 0
        for item in group_iter:
            assert item.device == device

            item_cpu = item.item()
            assert item_cpu == total_counter
            total_counter += 1

            group_size += 1
        group_sizes.append(group_size)

    assert group_sizes == expected_groups


@pytest.mark.local
def test_stateful_loader_checkpointing(tmp_path):
    dataset = SimpleDataset(size=10)
    device = torch.device("cpu")
    dp_rank = 3
    group_size = 2
    batch_size = 1

    loader = StatefulDataLoaderDataParallelAware(
        dataset,
        dp_rank=dp_rank,
        device=device,
        group_size=group_size,
        batch_size=batch_size,
        collate_fn=simple_collate,
        shuffle=False
    )

    loader_iter = iter(loader)

    group_gen = next(loader_iter)
    next(group_gen)
    next(group_gen)

    state = loader.state_dict()

    expected_key = f"dp_{dp_rank}"
    assert expected_key in state
    assert len(state) == 1

    # Create new loader
    loader_new = StatefulDataLoaderDataParallelAware(
        dataset,
        dp_rank=dp_rank,
        device=device,
        group_size=group_size,
        batch_size=batch_size,
        collate_fn=simple_collate,
        shuffle=False
    )

    loader_new.load_state_dict(state)

    new_iter = iter(loader_new)

    # Check first item of new iterator
    group_gen = next(new_iter)
    first_item = next(group_gen)  # This should be item index 2

    assert first_item.item() == 2.0


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("persistent_workers", [True, False])
@pytest.mark.distributed
def test_dataloader_e2e_integration(dist_ctx_dpr8, num_workers, pin_memory, persistent_workers):
    batch_maths = MockBatchMaths(
        data_loader_batch_size=2,
        num_microbatches_gradient_accumulation=2
    )
    loading_config = DataLoadingConfig(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    def provider(ctx: InitializeDatasetContext) -> InitializeDatasetResult:
        return InitializeDatasetResult(
            dataset=SimpleDataset(64),
            collator=simple_collate
        )

    factory = DataLoaderFactory(
        dist_context=dist_ctx_dpr8,
        provider=provider,
        config_data_loading=loading_config,
        batch_maths=batch_maths
    )

    loader = factory.build_dataloader_for_train_job()
    assert loader.batch_size == 2
    assert loader.pin_memory == pin_memory
    assert loader.persistent_workers == persistent_workers
    iter_loader = iter(loader)

    group_gen = next(iter_loader)
    batches = list(group_gen)

    assert len(batches) == 2  # group_size
    assert batches[0].is_cuda
    assert_close(batches[0], torch.tensor([0., 1.], device="cuda"))
