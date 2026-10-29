from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Self, TypedDict, cast

from mlflow import MlflowClient
from mlflow.entities import Param, RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE

from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig, TrackerPoint
from d9d.tracker.provider.flat import flatten_hparams, series_key
from d9d.tracker.value import Scalar, TrackerValue

from .config import MLflowConfig


class MLflowState(TypedDict):
    """Format of the saved MLflow tracker state."""

    run_id: str | None


class MLflowRun(BaseTrackerRun):
    """Tracking run that logs to an MLflow run.

    It does not write histograms, because MLflow has no histogram type.
    """

    def __init__(self, client: MlflowClient, run_id: str):
        """Constructs the ``MLflowRun`` object.

        Args:
            client: The MLflow client.
            run_id: The ID of the MLflow run to log to.
        """
        self._client = client
        self._run_id = run_id

    def write(self, point: TrackerPoint, value: TrackerValue):
        match value:
            case Scalar():
                # A synchronous write waits for one request to the server. An asynchronous write is sent in a batch
                # by a background thread.
                self._client.log_metric(
                    self._run_id, series_key(point), float(value.value), step=point.step, synchronous=False
                )
            case _:
                # MLflow has no histogram type, so the run writes only scalars.
                pass


class MLflowTracker(BaseTracker[MLflowConfig]):
    """Tracker that logs to MLflow.

    It saves the run ID in its state, so a job restored from a checkpoint continues the same run.
    """

    def __init__(self, config: MLflowConfig):
        """Constructs the ``MLflowTracker`` object.

        Args:
            config: The MLflow configuration.
        """
        self._config = config

        self._run_id: str | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = cast(MLflowState, state_dict)
        self._run_id = state["run_id"]

    def state_dict(self) -> dict[str, Any]:
        return {"run_id": self._run_id}

    def _create_run(self, client: MlflowClient, properties: RunConfig) -> str:
        experiment = client.get_experiment_by_name(self._config.experiment)
        if experiment is None:
            experiment_id = client.create_experiment(self._config.experiment)
        else:
            experiment_id = experiment.experiment_id

        tags = {MLFLOW_RUN_NOTE: properties.description} if properties.description is not None else None
        run_id = client.create_run(experiment_id, run_name=properties.name, tags=tags).info.run_id

        params = [Param(key, str(value)) for key, value in flatten_hparams(properties.hparams).items()]
        client.log_batch(run_id, params=params)
        return run_id

    @contextmanager
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        client = MlflowClient(tracking_uri=self._config.tracking_uri)

        if self._run_id is None:
            self._run_id = self._create_run(client, properties)
        else:
            # MLflow raises if a logged param gets another value, so a resumed run keeps the params of its start.
            client.update_run(self._run_id, status=RunStatus.to_string(RunStatus.RUNNING))

        try:
            yield MLflowRun(client, self._run_id)
        except BaseException:
            client.set_terminated(self._run_id, status=RunStatus.to_string(RunStatus.FAILED))
            raise
        # It also waits for the asynchronous writes.
        client.set_terminated(self._run_id)

    @classmethod
    def from_config(cls, config: MLflowConfig) -> Self:
        return cls(config)
