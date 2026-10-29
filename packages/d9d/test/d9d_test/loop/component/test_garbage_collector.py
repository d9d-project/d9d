import gc
from unittest.mock import MagicMock

import pytest
from d9d.core.dist_context import DeviceMeshParameters
from d9d.loop.component.garbage_collector import ManualGarbageCollector
from d9d.loop.component.job_schedule import JobSchedule
from d9d.loop.config import GarbageCollectionConfig, JobScheduleConfig, StepActionSpecial


class _FakeStream:
    @property
    def total_steps(self) -> int | None:
        return None


def _schedule_at_step(current_step: int) -> JobSchedule:
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=100), stream=_FakeStream())
    schedule._current_step = current_step
    return schedule


@pytest.fixture
def mock_gc(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(gc, "collect", mock.collect)
    monkeypatch.setattr(gc, "disable", mock.disable)
    monkeypatch.setattr(gc, "enable", mock.enable)
    return mock


@pytest.fixture
def gc_config_periodic():
    return GarbageCollectionConfig(period_steps=10)


@pytest.fixture
def gc_config_disable():
    return GarbageCollectionConfig(period_steps=StepActionSpecial.disable)


@pytest.mark.local
def test_manual_gc_context_manager_lifecycle(mock_gc, dist_ctx_factory, gc_config_periodic):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    manager = ManualGarbageCollector(dist_ctx, gc_config_periodic, _schedule_at_step(0))

    with manager:
        mock_gc.disable.assert_called_once()
        mock_gc.collect.assert_called_with(2)
        mock_gc.collect.reset_mock()
        mock_gc.disable.reset_mock()

    mock_gc.enable.assert_called_once()
    mock_gc.collect.assert_called_with(2)


@pytest.mark.local
def test_manual_gc_collect_forced(mock_gc, dist_ctx_factory, gc_config_periodic):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    manager = ManualGarbageCollector(dist_ctx, gc_config_periodic, _schedule_at_step(5))

    manager.collect_forced()

    mock_gc.collect.assert_called_once_with(2)


@pytest.mark.local
@pytest.mark.parametrize(
    ("current_step", "period", "should_collect"),
    [
        (8, 10, False),  # (8+1) % 10 = 9 → False
        (9, 10, True),  # (9+1) % 10 = 0 → True
        (10, 10, False),  # (10+1) % 10 = 1 → False
        (19, 10, True),  # (19+1) % 10 = 0 → True
    ],
)
def test_manual_gc_collect_periodic(mock_gc, dist_ctx_factory, current_step, period, should_collect):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    config = GarbageCollectionConfig(period_steps=period)
    manager = ManualGarbageCollector(dist_ctx, config, _schedule_at_step(current_step))
    manager.collect_periodic()

    if should_collect:
        mock_gc.collect.assert_called_once_with(1)
    else:
        mock_gc.collect.assert_not_called()


@pytest.mark.local
def test_manual_gc_disabled_config(mock_gc, dist_ctx_factory, gc_config_disable):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    schedule = _schedule_at_step(10)

    manager = ManualGarbageCollector(dist_ctx, gc_config_disable, schedule)

    manager.collect_periodic()
    mock_gc.collect.assert_not_called()
