import logging

import pytest
import torch
from d9d.tracker import RunConfig, tracker_from_config
from d9d.tracker.provider.log import LogRun, LogTracker, LogTrackerConfig


@pytest.mark.local
def test_log_run_writes_each_scalar(caplog):
    run = LogRun(logging.getLogger("test_log"))

    with caplog.at_level(logging.INFO, logger="test_log"):
        run.set_context({"stage": "train", "split": "train"})
        run.set_step(3)
        run.scalar("loss", 0.5)
        run.scalar("accuracy", 0.65, context={"split": "eval"})
        run.bins("hist", torch.randn(10))

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
        run.set_step(0)
        run.scalar("loss", 1.5)

    assert [(record.name, record.message) for record in caplog.records] == [("d9d", "step 0: loss=1.5")]
