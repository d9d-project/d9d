import pytest
import torch
from d9d.tracker import BaseTrackerRun, RunConfig, tracker_from_config
from d9d.tracker.provider.null import NullTracker, NullTrackerConfig


@pytest.mark.local
def test_null_tracker_lifecycle():
    tracker = NullTracker()

    assert tracker.state_dict() == {}
    tracker.load_state_dict({"random": "stuff"})

    run_config = RunConfig(name="test", description="desc")
    with tracker.open(run_config) as run:
        assert isinstance(run, BaseTrackerRun)
        run.set_step(10)
        run.set_context({"a": "b"})
        run.scalar("loss", 0.5)
        run.bins("hist", torch.randn(10))


@pytest.mark.local
def test_factory_null():
    config = NullTrackerConfig()
    tracker = tracker_from_config(config)
    assert isinstance(tracker, NullTracker)
