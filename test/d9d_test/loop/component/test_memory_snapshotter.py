import json
import tarfile
from pathlib import Path

import pytest
import torch
from d9d.core.dist_context import DeviceMeshParameters
from d9d.loop.component.job_schedule import JobSchedule
from d9d.loop.component.memory_snapshotter import ConfigurationMemorySnapshotter, JobMemorySnapshotter
from d9d.loop.config import JobScheduleConfig, MemorySnapshotConfig, MemorySnapshotStepsConfig
from pydantic import ValidationError


class _FakeStream:
    @property
    def total_steps(self) -> int | None:
        return None


@pytest.fixture(autouse=True)
def _stop_memory_history():
    yield
    torch.cuda.memory._record_memory_history(enabled=None)


def _schedule_at_step(current_step: int, total_steps: int) -> JobSchedule:
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=total_steps), stream=_FakeStream())
    schedule._current_step = current_step
    return schedule


def _config(tmp_path: Path, steps: MemorySnapshotStepsConfig | None, configure: bool = False) -> MemorySnapshotConfig:
    return MemorySnapshotConfig(
        enabled=True, snapshots_dir=tmp_path, max_entries=10000, configure=configure, steps=steps
    )


def _step_bytes(step: int) -> int:
    # distinct per step and aligned to the allocator rounding
    return (step + 1) * 2048


def _run_step(step: int):
    tensor = torch.empty(_step_bytes(step) // 4, device="cuda")
    del tensor


def _recorded_steps(tag_dir: Path, total_steps: int) -> list[int]:
    with tarfile.open(tag_dir / "memory.tar.gz", "r:gz") as tar:
        member = tar.extractfile("memory.json")
        assert member is not None
        snapshot = json.load(member)

    sizes = {entry["size"] for trace in snapshot["device_traces"] for entry in trace if entry["action"] == "alloc"}
    return [step for step in range(total_steps) if _step_bytes(step) in sizes]


@pytest.mark.local
@pytest.mark.parametrize(
    ("period_steps", "active_steps", "start_step", "total_steps", "expected"),
    [
        (4, 1, 0, 8, {"step_1": [0], "step_5": [4]}),
        (4, 2, 0, 8, {"step_2": [0, 1], "step_6": [4, 5]}),
        (4, 1, 2, 8, {"step_5": [4]}),
        # window interrupted by the end of the loop
        (4, 4, 0, 6, {"step_4": [0, 1, 2, 3], "step_6": [4, 5]}),
        # resumed inside a window
        (4, 2, 5, 8, {"step_6": [5]}),
        # no recording is started past the last step
        (4, 1, 0, 4, {"step_1": [0]}),
    ],
)
def test_job_snapshots_windows(
    dist_ctx_factory, tmp_path, period_steps, active_steps, start_step, total_steps, expected
):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    schedule = _schedule_at_step(start_step, total_steps)
    config = _config(tmp_path, MemorySnapshotStepsConfig(period_steps=period_steps, active_steps=active_steps))
    snapshotter = JobMemorySnapshotter(dist_context=dist_ctx, config=config, schedule=schedule)

    with snapshotter.open():
        for step in range(start_step, total_steps):
            _run_step(step)
            snapshotter.step()
            schedule.step()

    assert not torch._C._cuda_isHistoryEnabled()
    assert {path.name for path in tmp_path.iterdir()} == set(expected)
    for tag, steps in expected.items():
        assert _recorded_steps(tmp_path / tag, total_steps) == steps


@pytest.mark.local
def test_job_snapshots_dump_on_exception(dist_ctx_factory, tmp_path):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    schedule = _schedule_at_step(0, 8)
    config = _config(tmp_path, MemorySnapshotStepsConfig(period_steps=4, active_steps=2))
    snapshotter = JobMemorySnapshotter(dist_context=dist_ctx, config=config, schedule=schedule)

    with pytest.raises(ValueError, match="boom"), snapshotter.open():
        _run_step(0)
        snapshotter.step()
        schedule.step()

        _run_step(1)
        raise ValueError("boom")

    assert not torch._C._cuda_isHistoryEnabled()
    assert {path.name for path in tmp_path.iterdir()} == {"step_1"}
    assert _recorded_steps(tmp_path / "step_1", 8) == [0, 1]


@pytest.mark.local
@pytest.mark.parametrize("enabled", [True, False])
def test_job_snapshots_lifecycle(dist_ctx_factory, tmp_path, enabled):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    config = _config(tmp_path, MemorySnapshotStepsConfig(period_steps=4, active_steps=1)) if enabled else None
    snapshotter = JobMemorySnapshotter(dist_context=dist_ctx, config=config, schedule=_schedule_at_step(1, 8))

    with pytest.raises(RuntimeError, match="must be open"):
        snapshotter.step()

    with snapshotter.open():
        with pytest.raises(RuntimeError, match="already open"), snapshotter.open():
            pass
        snapshotter.step()


@pytest.mark.local
@pytest.mark.parametrize(
    "config",
    [
        None,
        MemorySnapshotConfig(
            enabled=False,
            snapshots_dir=Path("unused"),
            max_entries=10000,
            configure=True,
            steps=MemorySnapshotStepsConfig(period_steps=1, active_steps=1),
        ),
        MemorySnapshotConfig(enabled=True, snapshots_dir=Path("unused"), max_entries=10000, configure=True, steps=None),
    ],
)
def test_job_snapshots_disabled(dist_ctx_factory, tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    schedule = _schedule_at_step(0, 4)
    snapshotter = JobMemorySnapshotter(dist_context=dist_ctx, config=config, schedule=schedule)

    with snapshotter.open():
        for step in range(4):
            _run_step(step)
            assert not torch._C._cuda_isHistoryEnabled()
            snapshotter.step()
            schedule.step()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.local
def test_configuration_snapshot(dist_ctx_factory, tmp_path):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    config = _config(tmp_path, steps=None, configure=True)

    with ConfigurationMemorySnapshotter(dist_context=dist_ctx, config=config).record():
        _run_step(0)

    assert not torch._C._cuda_isHistoryEnabled()
    assert {path.name for path in tmp_path.iterdir()} == {"configure"}
    assert _recorded_steps(tmp_path / "configure", 1) == [0]


@pytest.mark.local
def test_configuration_snapshot_on_exception(dist_ctx_factory, tmp_path):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    config = _config(tmp_path, steps=None, configure=True)

    with pytest.raises(ValueError, match="boom"), ConfigurationMemorySnapshotter(dist_ctx, config).record():
        _run_step(0)
        raise ValueError("boom")

    assert _recorded_steps(tmp_path / "configure", 1) == [0]


@pytest.mark.local
@pytest.mark.parametrize(
    "config",
    [
        None,
        MemorySnapshotConfig(
            enabled=False, snapshots_dir=Path("unused"), max_entries=10000, configure=True, steps=None
        ),
        MemorySnapshotConfig(
            enabled=True,
            snapshots_dir=Path("unused"),
            max_entries=10000,
            configure=False,
            steps=MemorySnapshotStepsConfig(period_steps=1, active_steps=1),
        ),
    ],
)
def test_configuration_snapshot_disabled(dist_ctx_factory, tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    with ConfigurationMemorySnapshotter(dist_context=dist_ctx, config=config).record():
        assert not torch._C._cuda_isHistoryEnabled()
        _run_step(0)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.local
@pytest.mark.parametrize(
    ("period_steps", "active_steps"),
    [(4, 5), (0, 1), (4, 0)],
)
def test_steps_config_validation(period_steps, active_steps):
    with pytest.raises(ValidationError):
        MemorySnapshotStepsConfig(period_steps=period_steps, active_steps=active_steps)
