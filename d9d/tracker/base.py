import abc
from contextlib import contextmanager
from typing import TypeVar, Generic, Self, Generator, Any

import torch
from pydantic import BaseModel, Field
from torch.distributed.checkpoint.stateful import Stateful


class BaseTrackerRun(abc.ABC):
    """
    Abstract base class representing an active tracking session (run).

    This object is responsible for the actual logging of metrics, parameters,
    during train or inference run.
    """

    @abc.abstractmethod
    def set_step(self, step: int):
        """
        Updates the global step counter for subsequent logs.

        Args:
            step: The current step index (e.g., iteration number).
        """
        ...

    @abc.abstractmethod
    def set_context(self, context: dict[str, str]):
        """
        Sets a persistent context dictionary for subsequent logs.

        These context values (tags) will be attached to every metric logged
        until changed.

        Args:
            context: A dictionary of tag names and values.
        """
        ...

    @abc.abstractmethod
    def scalar(self, name: str, value: float, context: dict[str, str] | None = None):
        """
        Logs a scalar value.

        Args:
            name: The name of the metric.
            value: The scalar value to log.
            context: Optional ephemeral context specific to this metric event.
                Merged with global context if present.
        """
        ...

    @abc.abstractmethod
    def bins(self, name: str, values: torch.Tensor, context: dict[str, str] | None = None):
        """
        Logs a distribution/histogram of values.

        Args:
            name: The name of the metric.
            values: A tensor containing the population of values to bin.
            context: Optional ephemeral context specific to this metric event.
                Merged with global context if present.
        """
        ...


class RunConfig(BaseModel):
    """
    Configuration for initializing a specific logged run.

    Attributes:
        name: The display name of the experiment.
        description: An optional description of the experiment.
        hparams: A dictionary of hyperparameters to log at the start of the run.
    """

    name: str
    description: str | None
    hparams: dict[str, Any] = Field(default_factory=dict)


TConfig = TypeVar("TConfig", bound=BaseModel)


class BaseTracker(abc.ABC, Stateful, Generic[TConfig]):
    """
    Abstract base class for a tracker backend factory.

    This class manages the lifecycle of runs and integration with the
    distributed checkpointing system to ensure experiment continuity
    (e.g., resuming the same run hash after a restart).
    """

    @contextmanager
    @abc.abstractmethod
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        """
        Context manager that initiates and manages an experiment run.

        Args:
            properties: Configuration metadata for the run.

        Yields:
            An active BaseTrackerRun instance for logging metrics.
        """

        ...

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: TConfig) -> Self:
        """
        Factory method to create a tracker instance from a configuration object.

        Args:
            config: The backend-specific configuration object.

        Returns:
            An initialized instance of the tracker.
        """

        ...
