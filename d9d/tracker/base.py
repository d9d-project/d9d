import abc
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Generic, Self, TypeVar

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.distributed.checkpoint.stateful import Stateful


class BaseTrackerRun(abc.ABC):
    """Abstract base class for an active tracking session (run).

    A run logs metrics during training or inference.
    """

    @abc.abstractmethod
    def set_step(self, step: int):
        """Sets the global step for subsequent logs.

        Args:
            step: The current step, e.g. the iteration number.
        """
        ...

    @abc.abstractmethod
    def set_context(self, context: dict[str, str]):
        """Sets the context for subsequent logs.

        The context values (tags) are attached to every logged metric until the next call.

        Args:
            context: A dict of tag names and values.
        """
        ...

    @abc.abstractmethod
    def scalar(self, name: str, value: float, context: dict[str, str] | None = None):
        """Logs a scalar value.

        Args:
            name: The name of the metric.
            value: The scalar value to log.
            context: An optional context for this event only. It is merged with the run context.
        """
        ...

    @abc.abstractmethod
    def bins(self, name: str, values: torch.Tensor, context: dict[str, str] | None = None):
        """Logs a distribution (histogram) of values.

        Args:
            name: The name of the metric.
            values: The population of values to bin.
            context: An optional context for this event only. It is merged with the run context.
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
