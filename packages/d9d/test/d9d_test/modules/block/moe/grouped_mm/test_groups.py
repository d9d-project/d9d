import pytest
import torch
from d9d.module.block.moe.grouped_mm import ExpertTokenGroups


@pytest.mark.local
def test_offsets():
    groups = ExpertTokenGroups(torch.tensor([3, 0, 5], device="cuda"))

    assert groups.offsets.device.type == "cuda"
    assert groups.offsets.dtype == torch.int32
    assert groups.offsets.tolist() == [0, 3, 3, 8]


@pytest.mark.local
def test_host_counts_from_producer():
    host_counts = torch.tensor([3, 0, 5])
    groups = ExpertTokenGroups(host_counts.cuda(), host_counts=host_counts)

    assert groups.host_counts() is host_counts


@pytest.mark.local
def test_host_counts_copied_once():
    groups = ExpertTokenGroups(torch.tensor([3, 0, 5], device="cuda"))

    host_counts = groups.host_counts()

    assert host_counts.device == torch.device("cpu")
    assert host_counts.tolist() == [3, 0, 5]
    assert groups.host_counts() is host_counts
