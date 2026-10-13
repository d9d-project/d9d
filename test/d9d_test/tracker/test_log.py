import logging

import pytest
import torch
from d9d.tracker import Histogram, RunConfig, Scalar, TrackerPoint, tracker_from_config
from d9d.tracker.provider.log import LogRun, LogTracker, LogTrackerConfig


@pytest.mark.local
def test_log_run_writes_each_scalar(caplog):
    run = LogRun(logging.getLogger("test_log"))

    with caplog.at_level(logging.INFO, logger="test_log"):
        run.write(
            TrackerPoint(name="loss", step=3, context={"stage": "train", "split": "train"}),
            Scalar(value=torch.tensor(0.5)),
        )
        run.write(
            TrackerPoint(name="accuracy", step=3, context={"stage": "train", "split": "eval"}),
            Scalar(value=torch.tensor(0.65, dtype=torch.float64)),
        )
        run.write(TrackerPoint(name="hist", step=3, context={}), Histogram(counts=torch.randn(10)))

    assert caplog.messages == [
        "step 3 (stage=train, split=train): loss=0.5",
        "step 3 (stage=train, split=eval): accuracy=0.65",
    ]


@pytest.mark.local
def test_log_tracker_writes_to_d9d_logger(caplog):
    tracker = tracker_from_config(LogTrackerConfig())
    assert isinstance(tracker, LogTracker)
    assert tracker.state_dict() == {}

    with caplog.at_level(logging.INFO, logger="d9d"), tracker.open(RunConfig(name="test", description=None)) as run:
        run.write(TrackerPoint(name="loss", step=0, context={}), Scalar(value=torch.tensor(1.5)))

    assert [(record.name, record.message) for record in caplog.records] == [("d9d", "step 0: loss=1.5")]
