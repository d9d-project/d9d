from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal, Self

import torch
from pydantic import BaseModel

from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig


class NullTrackerConfig(BaseModel):
    """Configuration of the null tracker, which logs nothing.

    Attributes:
        provider: The discriminator field. Must be ``"null"``.
    """

    provider: Literal["null"] = "null"


class NullRun(BaseTrackerRun):
    """Tracking run that discards all inputs.

    Use it in tests or when tracking is disabled.
    """

    def set_step(self, step: int):
        pass

    def set_context(self, context: dict[str, str]):
        pass

    def scalar(self, name: str, value: float, context: dict[str, str] | None = None):
        pass

    def bins(self, name: str, values: torch.Tensor, context: dict[str, str] | None = None):
        pass


class NullTracker(BaseTracker[NullTrackerConfig]):
    """Tracker that opens ``NullRun`` runs.

    It keeps no state and does no I/O.
    """

    @contextmanager
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        yield NullRun()

    @classmethod
    def from_config(cls, config: NullTrackerConfig) -> Self:
        return cls()

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
