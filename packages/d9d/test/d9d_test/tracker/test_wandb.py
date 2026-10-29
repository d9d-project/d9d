from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from d9d.tracker import Histogram, RunConfig, Scalar, TrackerPoint, tracker_from_config
from d9d.tracker.provider.wandb.config import WandbConfig
from d9d.tracker.provider.wandb.tracker import WandbTracker


@pytest.fixture
def mock_wandb_init():
    with patch("d9d.tracker.provider.wandb.tracker.wandb.init") as mock_init:
        mock_run = MagicMock()
        mock_run.id = "run_id_123"
        mock_init.return_value = mock_run
        yield mock_init, mock_run


@pytest.mark.local
def test_wandb_tracker_lifecycle(mock_wandb_init):
    mock_init, mock_run = mock_wandb_init

    tracker = tracker_from_config(WandbConfig(project="proj", tags=["a"]))
    assert isinstance(tracker, WandbTracker)

    with tracker.open(RunConfig(name="test_run", description="desc", hparams={"lr": 0.01})) as run:
        kwargs = mock_init.call_args.kwargs
        assert kwargs["project"] == "proj"
        assert kwargs["tags"] == ["a"]
        assert kwargs["id"] is None
        assert kwargs["resume"] is None
        assert kwargs["name"] == "test_run"
        assert kwargs["notes"] == "desc"
        assert kwargs["config"] == {"lr": 0.01}
        assert kwargs["settings"].console == "off"

        run.write(TrackerPoint(name="loss", step=5, context={"stage": "train"}), Scalar(value=torch.tensor(0.5)))
        mock_run.log.assert_called_with({"train/loss": 0.5}, step=5)

        run.write(TrackerPoint(name="weights", step=5, context={}), Histogram(counts=torch.tensor([1.0, 0.0, 3.0])))
        (logged,), _ = mock_run.log.call_args
        assert logged["weights"].histogram == [1.0, 0.0, 3.0]
        assert logged["weights"].bins == [0, 1, 2, 3]

    mock_run.finish.assert_called_once_with()
    assert tracker.state_dict() == {"run_id": "run_id_123"}


@pytest.mark.local
def test_wandb_tracker_resume(mock_wandb_init):
    mock_init, _ = mock_wandb_init

    tracker = WandbTracker(WandbConfig(project="proj"))
    tracker.load_state_dict({"run_id": "run_from_prev_job"})

    with tracker.open(RunConfig(name="test", description=None)):
        assert mock_init.call_args.kwargs["id"] == "run_from_prev_job"
        assert mock_init.call_args.kwargs["resume"] == "must"


@pytest.mark.local
def test_wandb_tracker_marks_failed_run(mock_wandb_init):
    _, mock_run = mock_wandb_init

    tracker = WandbTracker(WandbConfig(project="proj"))
    with pytest.raises(RuntimeError), tracker.open(RunConfig(name="test", description=None)):
        raise RuntimeError("crash")

    mock_run.finish.assert_called_once_with(exit_code=1)


@pytest.mark.local
def test_wandb_histogram_limit(mock_wandb_init):
    tracker = WandbTracker(WandbConfig(project="proj"))

    with tracker.open(RunConfig(name="test", description=None)) as run, pytest.raises(ValueError, match="512"):
        run.write(TrackerPoint(name="weights", step=0, context={}), Histogram(counts=torch.from_numpy(np.ones(513))))
