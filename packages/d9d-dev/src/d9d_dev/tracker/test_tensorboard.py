import pytest
import torch
from d9d.tracker import Histogram, RunConfig, Scalar, TrackerPoint, tracker_from_config
from d9d.tracker.provider.tensorboard.config import TensorBoardConfig
from d9d.tracker.provider.tensorboard.tracker import TensorBoardTracker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _read(log_dir):
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0, "histograms": 0, "tensors": 0})
    events.Reload()
    return events


def _train_point(name, step):
    return TrackerPoint(name=name, step=step, context={"stage": "train"})


@pytest.mark.local
def test_tensorboard_writes_values(tmp_path):
    tracker = tracker_from_config(TensorBoardConfig(log_dir=str(tmp_path)))
    assert isinstance(tracker, TensorBoardTracker)

    run_config = RunConfig(name="test", description="desc", hparams={"run": {"lr": 0.01}, "params": {"a|b": "c|d"}})
    with tracker.open(run_config) as run:
        run.write(_train_point("loss", 0), Scalar(value=torch.tensor(1.5)))
        run.write(_train_point("loss", 1), Scalar(value=torch.tensor(0.5, dtype=torch.bfloat16)))
        run.write(_train_point("weights", 1), Histogram(counts=torch.tensor([1.0, 0.0, 3.0])))

    events = _read(tmp_path)
    assert [(e.step, e.value) for e in events.Scalars("train/loss")] == [(0, 1.5), (1, 0.5)]

    (histogram,) = events.Histograms("train/weights")
    assert histogram.step == 1
    assert histogram.histogram_value.bucket_limit == [1.0, 2.0, 3.0]
    assert histogram.histogram_value.bucket == [1.0, 0.0, 3.0]

    (summary,) = events.Tensors("run/text_summary")
    assert summary.tensor_proto.string_val[0].decode() == (
        "# test\n\ndesc\n\n| Hyperparameter | Value |\n|:--|:--|\n| params/a|b | c\\|d |\n| run/lr | 0.01 |"
    )
    assert tracker.state_dict() == {"last_step": 1}


@pytest.mark.local
def test_tensorboard_resume_drops_steps_after_checkpoint(tmp_path):
    config = TensorBoardConfig(log_dir=str(tmp_path))
    run_config = RunConfig(name="test", description=None)

    crashed = TensorBoardTracker(config)
    with crashed.open(run_config) as run:
        for step in range(3):
            run.write(_train_point("loss", step), Scalar(value=float(step)))
        checkpoint = crashed.state_dict()
        # The job crashes after it has logged steps after the checkpoint.
        for step in range(3, 5):
            run.write(_train_point("loss", step), Scalar(value=float(step)))
    assert checkpoint == {"last_step": 2}

    resumed = TensorBoardTracker(config)
    resumed.load_state_dict(checkpoint)
    with resumed.open(run_config) as run:
        run.write(_train_point("loss", 3), Scalar(value=30.0))

    events = _read(tmp_path)
    assert [(e.step, e.value) for e in events.Scalars("train/loss")] == [(0, 0.0), (1, 1.0), (2, 2.0), (3, 30.0)]
    # The resumed job does not write the run summary again.
    assert len(events.Tensors("run/text_summary")) == 1
