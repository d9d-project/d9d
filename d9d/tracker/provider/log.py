import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict

from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig


class LogTrackerConfig(BaseModel):
    """Configuration for the log tracker, which writes scalars to the Python logger of d9d.

    Attributes:
        provider: Discriminator field. Always ``"log"``.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["log"] = "log"


def _format_line(step: int, context: dict[str, str], name: str, value: float) -> str:
    tags = ", ".join(f"{key}={tag}" for key, tag in context.items())
    prefix = f"step {step} ({tags})" if tags else f"step {step}"
    return f"{prefix}: {name}={value:.6g}"


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
        self._step = 0
        self._context: dict[str, str] = {}

    def set_step(self, step: int):
        self._step = step

    def set_context(self, context: dict[str, str]):
        self._context = context

    def scalar(self, name: str, value: float, context: dict[str, str] | None = None):
        # The context of the event overrides the context of the run.
        merged_context = self._context if context is None else {**self._context, **context}
        self._logger.info(_format_line(self._step, merged_context, name, value))

    def bins(self, name: str, values: torch.Tensor, context: dict[str, str] | None = None):
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
