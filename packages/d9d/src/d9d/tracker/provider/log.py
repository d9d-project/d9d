import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict

from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig, TrackerPoint
from d9d.tracker.value import Scalar, TrackerValue


class LogTrackerConfig(BaseModel):
    """Configuration for the log tracker, which writes scalars to the Python logger of d9d.

    Attributes:
        provider: Discriminator field. Always ``"log"``.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["log"] = "log"


def _format_line(point: TrackerPoint, value: float) -> str:
    tags = ", ".join(f"{key}={tag}" for key, tag in point.context.items())
    prefix = f"step {point.step} ({tags})" if tags else f"step {point.step}"
    return f"{prefix}: {point.name}={value:.6g}"


class LogRun(BaseTrackerRun):
    """Tracking run that writes each scalar as one log line.

    It does not write distributions.
    """

    def __init__(self, logger: logging.Logger):
        """Constructs the ``LogRun`` object.

        Args:
            logger: The logger to write to.
        """
        self._logger = logger

    def write(self, point: TrackerPoint, value: TrackerValue):
        match value:
            case Scalar():
                self._logger.info(_format_line(point, float(value.value)))
            case _:
                # The log run writes only scalars, so it skips every other value type.
                pass


class LogTracker(BaseTracker[LogTrackerConfig]):
    """Tracker that writes scalars to the Python logger of d9d, named ``d9d``.

    It keeps no state, so a resumed job logs from the step it resumes at.
    """

    @contextmanager
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        # The DistributedContext configures the d9d logger with the rank prefix of each line.
        yield LogRun(logging.getLogger("d9d"))

    @classmethod
    def from_config(cls, config: LogTrackerConfig) -> Self:
        return cls()

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
