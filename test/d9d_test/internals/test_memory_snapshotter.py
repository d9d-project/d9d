import json
import tarfile
from pathlib import Path

import pytest
import torch
from d9d.core.dist_context import REGULAR_DOMAIN, DeviceMeshParameters
from d9d.internals.profiling import MemorySnapshotter
from torch.profiler import record_function


@pytest.fixture(autouse=True)
def _stop_memory_history():
    yield
    torch.cuda.memory._record_memory_history(enabled=None)


def _load_snapshot(tar_path: Path) -> dict:
    json_name = tar_path.name.removesuffix(".tar.gz") + ".json"
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.extractfile(json_name)
        assert member is not None
        return json.load(member)


def _alloc_sizes(snapshot: dict) -> set[int]:
    return {entry["size"] for trace in snapshot["device_traces"] for entry in trace if entry["action"] == "alloc"}


@pytest.mark.local
def test_record_dumps_snapshot(dist_ctx_factory, tmp_path):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    snapshotter = MemorySnapshotter(save_dir=tmp_path, max_entries=1000, dist_context=dist_ctx)

    with snapshotter.record("some_tag"), record_function("my_region"):
        tensor = torch.empty(512, device="cuda")
        del tensor

    assert not torch._C._cuda_isHistoryEnabled()
    assert not (tmp_path / "some_tag" / "memory.json").exists()

    snapshot = _load_snapshot(tmp_path / "some_tag" / "memory.tar.gz")

    assert 512 * 4 in _alloc_sizes(snapshot)
    assert "my_region" in {annotation["name"] for annotation in snapshot["external_annotations"]}


@pytest.mark.local
def test_record_starts_with_clear_history(dist_ctx_factory, tmp_path):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    snapshotter = MemorySnapshotter(save_dir=tmp_path, max_entries=1000, dist_context=dist_ctx)

    with snapshotter.record("first"):
        tensor = torch.empty(512, device="cuda")
        del tensor

    with snapshotter.record("second"):
        tensor = torch.empty(1024, device="cuda")
        del tensor

    assert _alloc_sizes(_load_snapshot(tmp_path / "second" / "memory.tar.gz")) == {1024 * 4}


@pytest.mark.local
def test_record_dumps_snapshot_on_exception(dist_ctx_factory, tmp_path):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    snapshotter = MemorySnapshotter(save_dir=tmp_path, max_entries=1000, dist_context=dist_ctx)

    with pytest.raises(ValueError, match="boom"), snapshotter.record("failed"):
        tensor = torch.empty(512, device="cuda")
        del tensor
        raise ValueError("boom")

    assert not torch._C._cuda_isHistoryEnabled()
    assert 512 * 4 in _alloc_sizes(_load_snapshot(tmp_path / "failed" / "memory.tar.gz"))


@pytest.mark.local
def test_start_while_recording_raises(dist_ctx_factory, tmp_path):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    snapshotter = MemorySnapshotter(save_dir=tmp_path, max_entries=1000, dist_context=dist_ctx)

    snapshotter.start()
    with pytest.raises(RuntimeError, match="already being recorded"):
        snapshotter.start()


@pytest.mark.local
def test_dump_without_recording_raises(dist_ctx_factory, tmp_path):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    snapshotter = MemorySnapshotter(save_dir=tmp_path, max_entries=1000, dist_context=dist_ctx)

    with pytest.raises(RuntimeError, match="not being recorded"):
        snapshotter.dump_and_stop("tag")

    assert not (tmp_path / "tag").exists()


@pytest.mark.distributed
def test_e2e(dist_ctx_factory, shared_tmp_dir):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    snapshotter = MemorySnapshotter(save_dir=shared_tmp_dir, max_entries=1000, dist_context=dist_ctx)

    with snapshotter.record("some_tag"):
        tensor = torch.empty(512, device="cuda")
        del tensor

    dist_ctx.wait_world()

    mesh = dist_ctx.mesh_for(REGULAR_DOMAIN)
    coord_str = "-".join(map(str, mesh.get_coordinate()))
    expected_filename_base = f"rank-{mesh.get_rank()}-coord-{coord_str}-memory"

    tag_dir = shared_tmp_dir / "some_tag"
    assert not (tag_dir / f"{expected_filename_base}.json").exists()
    assert 512 * 4 in _alloc_sizes(_load_snapshot(tag_dir / f"{expected_filename_base}.tar.gz"))
