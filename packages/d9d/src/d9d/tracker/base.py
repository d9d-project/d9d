import abc
import dataclasses
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from torch.distributed.checkpoint.stateful import Stateful

from .value import TrackerValue


@dataclasses.dataclass(frozen=True, kw_only=True)
class TrackerPoint:
    """Point of a tracked series that a value is written to.

    Attributes:
        name: The name of the series, e.g. ``"loss"``.
        step: The step the value belongs to.
        context: The tags of the value, e.g. ``{"stage": "train"}``.
    """

    name: str
    step: int
    context: dict[str, str]


class BaseTrackerRun(abc.ABC):
    """Abstract base class for an active tracking session (run) of a tracker backend.

    A run writes values to the backend. It keeps no step or context: every call carries them in a ``TrackerPoint``.
    Training code does not call it directly, but logs through a ``TrackerWriter``.
    """

    @abc.abstractmethod
    def write(self, point: TrackerPoint, value: TrackerValue):
        """Writes a value.

        A run raises a ``ValueError`` for a value type that it does not handle. A run that writes only some types on
        purpose can skip the others instead. Calls come in the order the values were logged.

        Args:
            point: The series, step and tags of the value.
            value: The value. Its tensors are on the CPU, in the dtype they were logged with.
        """
        ...


class RunConfig(BaseModel):
    """Configuration for a tracked run.

    Attributes:
        name: The display name of the experiment.
        description: An optional description of the experiment.
        hparams: The hyperparameters to log at the start of the run.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    hparams: dict[str, Any] = Field(default_factory=dict)


TConfig = TypeVar("TConfig", bound=BaseModel)


class BaseTracker(abc.ABC, Stateful, Generic[TConfig]):
    """Abstract base class for a tracker backend that opens runs.

    A tracker is ``Stateful``. A job restored from a checkpoint can then continue the same run, for example the
    same Aim run hash.
    """

    @contextmanager
    @abc.abstractmethod
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        """Opens a run for the duration of the block.

        Args:
            properties: The configuration of the run.

        Yields:
            The active run to log metrics to.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: TConfig) -> Self:
        """Creates a tracker from its configuration.

        Args:
            config: The backend-specific configuration.

        Returns:
            The tracker.
        """
        ...
