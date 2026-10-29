from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Self, TypedDict, cast

import torch
from torch.utils.tensorboard import SummaryWriter

from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig, TrackerPoint
from d9d.tracker.provider.flat import flatten_hparams, series_key
from d9d.tracker.value import Histogram, Scalar, TrackerValue

from .config import TensorBoardConfig


class TensorBoardState(TypedDict):
    """Format of the saved TensorBoard tracker state."""

    last_step: int | None


def _run_summary(properties: RunConfig) -> str:
    lines = [f"# {properties.name}", ""]
    if properties.description is not None:
        lines += [properties.description, ""]
    lines += ["| Hyperparameter | Value |", "|:--|:--|"]
    for key, value in flatten_hparams(properties.hparams).items():
        # A pipe in a value would end the table cell.
        escaped = str(value).replace("|", "\\|")
        lines.append(f"| {key} | {escaped} |")
    return "\n".join(lines)


class TensorBoardRun(BaseTrackerRun):
    """Tracking run that writes TensorBoard event files."""

    def __init__(self, writer: SummaryWriter, last_step: int | None):
        """Constructs the ``TensorBoardRun`` object.

        Args:
            writer: The writer of the event files.
            last_step: The last step that the run has written before, or ``None``.
        """
        self._writer = writer
        self._last_step = last_step

    @property
    def last_step(self) -> int | None:
        """The last step that the run has written, or ``None``."""
        return self._last_step

    def write(self, point: TrackerPoint, value: TrackerValue):
        key = series_key(point)

        match value:
            case Scalar():
                self._writer.add_scalar(key, float(value.value), global_step=point.step)
            case Histogram():
                counts = value.counts.double()
                num_bins = counts.shape[0]
                # A d9d histogram has no value range, so bin i spans [i, i + 1) and its values sit at its center.
                centers = torch.arange(num_bins, dtype=torch.float64) + 0.5
                self._writer.add_histogram_raw(
                    key,
                    min=0,
                    max=num_bins,
                    num=counts.sum().item(),
                    sum=(counts * centers).sum().item(),
                    sum_squares=(counts * centers**2).sum().item(),
                    bucket_limits=list(range(1, num_bins + 1)),
                    bucket_counts=counts.tolist(),
                    global_step=point.step,
                )
            case _:
                raise ValueError(f"Tracker value type ({type(value).__name__}) is not supported by TensorBoard.")

        self._last_step = point.step


class TensorBoardTracker(BaseTracker[TensorBoardConfig]):
    """Tracker that writes TensorBoard event files.

    It saves the last step that it has written in its state. A job restored from a checkpoint continues in the same
    directory, and TensorBoard drops the values that the previous job wrote after the checkpoint.
    """

    def __init__(self, config: TensorBoardConfig):
        """Constructs the ``TensorBoardTracker`` object.

        Args:
            config: The TensorBoard configuration.
        """
        self._config = config

        self._last_step: int | None = None
        self._run: TensorBoardRun | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = cast(TensorBoardState, state_dict)
        self._last_step = state["last_step"]

    def state_dict(self) -> dict[str, Any]:
        last_step = self._run.last_step if self._run is not None else self._last_step
        return {"last_step": last_step}

    @contextmanager
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        # The loop writes every value of the steps before a checkpoint before it saves the checkpoint, so a resumed
        # job writes the steps after last_step again.
        purge_step = self._last_step + 1 if self._last_step is not None else None
        writer = SummaryWriter(log_dir=self._config.log_dir, purge_step=purge_step)
        if purge_step is None:
            writer.add_text("run", _run_summary(properties), global_step=0)

        self._run = TensorBoardRun(writer, last_step=self._last_step)
        try:
            yield self._run
        finally:
            self._last_step = self._run.last_step
            self._run = None
            writer.close()

    @classmethod
    def from_config(cls, config: TensorBoardConfig) -> Self:
        return cls(config)
