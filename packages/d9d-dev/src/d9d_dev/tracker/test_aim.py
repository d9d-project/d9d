import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from d9d.tracker import Histogram, RunConfig, Scalar, TrackerPoint, tracker_from_config
from d9d.tracker.provider.aim.config import AimConfig
from d9d.tracker.provider.aim.tracker import AimTracker


@pytest.fixture
def mock_aim_run():
    with patch("d9d.tracker.provider.aim.tracker.Run") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.hash = "mocked_hash_123"
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


@pytest.fixture
def mock_aim_dist():
    """Mocks aim.Distribution."""
    with patch("d9d.tracker.provider.aim.tracker.Distribution") as mock_dist:
        yield mock_dist


@pytest.mark.local
def test_aim_tracker_lifecycle(mock_aim_run):
    mock_cls, mock_instance = mock_aim_run

    config = AimConfig(repo="/test_aim")
    tracker = AimTracker(config)

    run_props = RunConfig(name="test_run", description="test_desc", hparams={"lr": 0.01})

    with tracker.open(run_props):
        # Check initialization
        mock_cls.assert_called_with(
            run_hash=None,
            repo="/test_aim",
            log_system_params=True,
            capture_terminal_logs=True,
            system_tracking_interval=10,
        )

        # Check metadata
        assert mock_instance.name == "test_run"
        assert mock_instance.description == "test_desc"
        mock_instance.__setitem__.assert_called_with("hparams", {"lr": 0.01})

    mock_instance.close.assert_called_once()
    assert tracker.state_dict() == {"restart_hash": "mocked_hash_123"}


@pytest.mark.local
def test_aim_tracker_resume(mock_aim_run):
    mock_cls, _ = mock_aim_run

    tracker = AimTracker(AimConfig(repo="."))
    state_dict = {"restart_hash": "hash_from_prev_job"}
    tracker.load_state_dict(state_dict)

    with tracker.open(RunConfig(name="resumed_run", description=None)):
        mock_cls.assert_called_with(
            run_hash="hash_from_prev_job",
            repo=".",
            log_system_params=True,
            capture_terminal_logs=True,
            system_tracking_interval=10,
        )


@pytest.mark.local
def test_aim_run_logging(mock_aim_run, mock_aim_dist):
    _, mock_instance = mock_aim_run

    tracker = AimTracker(AimConfig(repo="."))

    with tracker.open(RunConfig(name="test", description=None)) as run:
        run.write(
            TrackerPoint(name="accuracy", step=100, context={"phase": "train"}),
            Scalar(value=torch.tensor(0.9, dtype=torch.float64)),
        )
        mock_instance.track.assert_called_with(name="accuracy", value=0.9, context={"phase": "train"}, step=100)

        # Bins/Histograms
        tensor_vals = torch.tensor([1.0, 2.0, 3.0])
        run.write(TrackerPoint(name="logits", step=100, context={"phase": "train"}), Histogram(counts=tensor_vals))

        # Verify tensor -> numpy conversion
        assert mock_aim_dist.called
        kwargs = mock_aim_dist.call_args[1]
        assert (kwargs["hist"] == tensor_vals.numpy()).all()


@pytest.mark.local
def test_factory_aim():
    with patch.dict(sys.modules, {"aim": MagicMock()}):
        config = AimConfig(repo="./test_repo")
        tracker = tracker_from_config(config)
        assert isinstance(tracker, AimTracker)
