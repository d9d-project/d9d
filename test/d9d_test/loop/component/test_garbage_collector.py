import gc
from unittest.mock import MagicMock

import pytest
from d9d.core.dist_context import DeviceMeshParameters
from d9d.loop.component.garbage_collector import ManualGarbageCollector
from d9d.loop.component.stepper import Stepper
from d9d.loop.config import GarbageCollectionConfig, StepActionSpecial


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

    manager = ManualGarbageCollector(dist_ctx, gc_config_periodic, Stepper(0, 100))

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

    manager = ManualGarbageCollector(dist_ctx, gc_config_periodic, Stepper(5, 100))

    manager.collect_forced()

    mock_gc.collect.assert_called_once_with(2)


@pytest.mark.local
@pytest.mark.parametrize(
    ("current_step", "period", "should_collect"),
    [
        (9, 10, False),  # 9 % 10 != 0
        (10, 10, True),  # 10 % 10 == 0
        (11, 10, False),
        (20, 10, True),
    ],
)
def test_manual_gc_collect_periodic(mock_gc, dist_ctx_factory, current_step, period, should_collect):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    config = GarbageCollectionConfig(period_steps=period)
    manager = ManualGarbageCollector(dist_ctx, config, Stepper(current_step, 100))
    manager.collect_periodic()

    if should_collect:
        mock_gc.collect.assert_called_once_with(1)
    else:
        mock_gc.collect.assert_not_called()


@pytest.mark.local
def test_manual_gc_disabled_config(mock_gc, dist_ctx_factory, gc_config_disable):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    stepper = Stepper(10, 100)

    manager = ManualGarbageCollector(dist_ctx, gc_config_disable, stepper)

    manager.collect_periodic()
    mock_gc.collect.assert_not_called()
