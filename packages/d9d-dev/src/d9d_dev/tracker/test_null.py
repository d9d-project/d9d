import pytest
import torch
from d9d.tracker import BaseTrackerRun, Histogram, RunConfig, Scalar, TrackerPoint, tracker_from_config
from d9d.tracker.provider.null import NullTracker, NullTrackerConfig


@pytest.mark.local
def test_null_tracker_lifecycle():
    tracker = NullTracker()

    assert tracker.state_dict() == {}
    tracker.load_state_dict({"random": "stuff"})

    run_config = RunConfig(name="test", description="desc")
    with tracker.open(run_config) as run:
        assert isinstance(run, BaseTrackerRun)
        run.write(TrackerPoint(name="loss", step=10, context={"a": "b"}), Scalar(value=torch.tensor(0.5)))
        run.write(TrackerPoint(name="hist", step=10, context={"a": "b"}), Histogram(counts=torch.randn(10)))


@pytest.mark.local
def test_factory_null():
    config = NullTrackerConfig()
    tracker = tracker_from_config(config)
    assert isinstance(tracker, NullTracker)
