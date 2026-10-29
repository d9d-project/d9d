from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Self, TypedDict, cast

import numpy as np
import wandb

from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig, TrackerPoint
from d9d.tracker.provider.flat import series_key
from d9d.tracker.value import Histogram, Scalar, TrackerValue

from .config import WandbConfig


class WandbState(TypedDict):
    """Format of the saved W&B tracker state."""

    run_id: str | None


class WandbRun(BaseTrackerRun):
    """Tracking run that logs to a ``wandb.Run``."""

    def __init__(self, run: wandb.Run):
        """Constructs the ``WandbRun`` object.

        Args:
            run: The W&B run to log to.
        """
        self._run = run

    def write(self, point: TrackerPoint, value: TrackerValue):
        match value:
            case Scalar():
                log_value = float(value.value)
            case Histogram():
                # NumPy has no bf16.
                counts = value.counts.float().numpy()
                # A d9d histogram has no value range, so bin i spans [i, i + 1).
                log_value = wandb.Histogram(np_histogram=(counts, np.arange(counts.shape[0] + 1)))
            case _:
                raise ValueError(f"Tracker value type ({type(value).__name__}) is not supported by W&B.")

        self._run.log({series_key(point): log_value}, step=point.step)


class WandbTracker(BaseTracker[WandbConfig]):
    """Tracker that logs to Weights & Biases.

    It saves the run ID in its state, so a job restored from a checkpoint continues the same run.
    """

    def __init__(self, config: WandbConfig):
        """Constructs the ``WandbTracker`` object.

        Args:
            config: The W&B configuration.
        """
        self._config = config

        self._run_id: str | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = cast(WandbState, state_dict)
        self._run_id = state["run_id"]

    def state_dict(self) -> dict[str, Any]:
        return {"run_id": self._run_id}

    @contextmanager
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        run = wandb.init(
            project=self._config.project,
            entity=self._config.entity,
            group=self._config.group,
            tags=self._config.tags,
            mode=self._config.mode,
            dir=self._config.dir,
            id=self._run_id,
            # Fail instead of starting a new run if the run of the checkpoint does not exist.
            resume="must" if self._run_id is not None else None,
            name=properties.name,
            notes=properties.description,
            config=properties.hparams,
            settings=wandb.Settings(console="auto" if self._config.capture_terminal_logs else "off"),
        )
        self._run_id = run.id

        try:
            yield WandbRun(run)
        except BaseException:
            run.finish(exit_code=1)
            raise
        run.finish()

    @classmethod
    def from_config(cls, config: WandbConfig) -> Self:
        return cls(config)
