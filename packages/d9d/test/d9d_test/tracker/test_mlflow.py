import pytest
import torch
from d9d.tracker import Histogram, RunConfig, Scalar, TrackerPoint, tracker_from_config
from d9d.tracker.provider.mlflow.config import MLflowConfig
from d9d.tracker.provider.mlflow.tracker import MLflowTracker
from mlflow import MlflowClient


@pytest.fixture
def config(tmp_path):
    return MLflowConfig(experiment="test", tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")


def _loss_history(client, run_id):
    return [(m.step, m.value) for m in client.get_metric_history(run_id, "train/loss")]


@pytest.mark.local
def test_mlflow_lifecycle(config):
    tracker = tracker_from_config(config)
    assert isinstance(tracker, MLflowTracker)

    run_config = RunConfig(name="test_run", description="desc", hparams={"run": {"lr": 0.01}, "params": {"a": None}})
    with tracker.open(run_config) as run:
        run.write(TrackerPoint(name="loss", step=0, context={"stage": "train"}), Scalar(value=torch.tensor(1.5)))
        run.write(TrackerPoint(name="loss", step=1, context={"stage": "train"}), Scalar(value=0.5))
        run.write(TrackerPoint(name="weights", step=1, context={}), Histogram(counts=torch.ones(3)))

    run_id = tracker.state_dict()["run_id"]
    mlflow_run = MlflowClient(config.tracking_uri).get_run(run_id)
    assert mlflow_run.info.run_name == "test_run"
    assert mlflow_run.info.status == "FINISHED"
    assert mlflow_run.data.tags["mlflow.note.content"] == "desc"
    assert mlflow_run.data.params == {"run/lr": "0.01", "params/a": "None"}
    assert mlflow_run.data.metrics.keys() == {"train/loss"}
    assert _loss_history(MlflowClient(config.tracking_uri), run_id) == [(0, 1.5), (1, 0.5)]


@pytest.mark.local
def test_mlflow_resume_keeps_run_and_params(config):
    first = MLflowTracker(config)
    with pytest.raises(RuntimeError), first.open(RunConfig(name="test", description=None, hparams={"lr": 0.01})) as run:
        run.write(TrackerPoint(name="loss", step=0, context={"stage": "train"}), Scalar(value=1.0))
        raise RuntimeError("crash")

    client = MlflowClient(config.tracking_uri)
    run_id = first.state_dict()["run_id"]
    assert client.get_run(run_id).info.status == "FAILED"

    resumed = MLflowTracker(config)
    resumed.load_state_dict(first.state_dict())
    # MLflow raises if a logged param gets another value, so the resumed run keeps the params of its start.
    with resumed.open(RunConfig(name="test", description=None, hparams={"lr": 0.02})) as run:
        run.write(TrackerPoint(name="loss", step=1, context={"stage": "train"}), Scalar(value=2.0))

    assert resumed.state_dict() == {"run_id": run_id}
    mlflow_run = client.get_run(run_id)
    assert mlflow_run.info.status == "FINISHED"
    assert mlflow_run.data.params == {"lr": "0.01"}
    assert _loss_history(client, run_id) == [(0, 1.0), (1, 2.0)]
