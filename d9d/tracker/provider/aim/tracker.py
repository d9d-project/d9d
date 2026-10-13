from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Self, TypedDict, cast

from aim import Distribution, Run

from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig, TrackerPoint
from d9d.tracker.value import Histogram, Scalar, TrackerValue

from .config import AimConfig


class AimState(TypedDict):
    """Format of the saved Aim tracker state."""

    restart_hash: str | None


class AimRun(BaseTrackerRun):
    """Tracking run that logs to an ``aim.Run``."""

    def __init__(self, run: Run):
        """Constructs the ``AimRun`` object.

        Args:
            run: The Aim run to log to.
        """
        self._run = run

    def write(self, point: TrackerPoint, value: TrackerValue):
        match value:
            case Scalar():
                track_value = float(value.value)
            case Histogram():
                # NumPy has no bf16.
                counts = value.counts.float().numpy()
                track_value = Distribution(hist=counts, bin_range=(0, counts.shape[0]))
            case _:
                raise ValueError(f"Tracker value type ({type(value).__name__}) is not supported by Aim.")

        self._run.track(name=point.name, value=track_value, context=point.context, step=point.step)


class AimTracker(BaseTracker[AimConfig]):
    """Tracker that logs to Aim.

    It saves the run hash in its state, so a job restored from a checkpoint continues the same run.
    """

    def __init__(self, config: AimConfig):
        """Constructs the ``AimTracker`` object.

        Args:
            config: The Aim configuration.
        """
        self._config = config

        self._restart_hash: str | None = None
        self._run: Run | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = cast(AimState, state_dict)
        self._restart_hash = state["restart_hash"]

    def state_dict(self) -> dict[str, Any]:
        return {"restart_hash": self._restart_hash}

    @contextmanager
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        run = Run(
            run_hash=self._restart_hash,
            repo=self._config.repo,
            log_system_params=self._config.log_system_params,
            capture_terminal_logs=self._config.capture_terminal_logs,
            system_tracking_interval=self._config.system_tracking_interval,
        )
        run.name = properties.name
        run.description = properties.description
        run["hparams"] = properties.hparams

        self._restart_hash = run.hash
        self._run = run

        yield AimRun(run)

        self._run.close()
        self._run = None

    @classmethod
    def from_config(cls, config: AimConfig) -> Self:
        return cls(config)
