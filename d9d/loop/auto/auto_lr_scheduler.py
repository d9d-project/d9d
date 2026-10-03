"""Learning rate scheduler provider built from a configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from d9d.core.protocol import LRSchedulerProtocol
from d9d.loop.control import InitializeLRSchedulerContext, LRSchedulerProvider
from d9d.lr_scheduler.piecewise import PiecewiseSchedulerConfig, piecewise_scheduler_from_config


class PiecewiseConfig(BaseModel):
    """Configuration for the piecewise learning rate scheduler.

    Attributes:
        name: Discriminator field. Always ``"piecewise"``.
        scheduler: Detailed configuration for the piecewise schedule.
    """

    name: Literal["piecewise"] = "piecewise"

    scheduler: PiecewiseSchedulerConfig


AutoLRSchedulerConfig = Annotated[PiecewiseConfig, Field(discriminator="name")]
"""Discriminated union of the supported learning rate scheduler configurations."""


class AutoLRSchedulerProvider(LRSchedulerProvider):
    """Builds a learning rate scheduler from a configuration object."""

    def __init__(self, config: AutoLRSchedulerConfig):
        """Constructs the ``AutoLRSchedulerProvider`` object."""
        self._config = config

    def __call__(self, context: InitializeLRSchedulerContext) -> LRSchedulerProtocol:
        match self._config:
            case PiecewiseConfig():
                return piecewise_scheduler_from_config(
                    self._config.scheduler, optimizer=context.optimizer, total_steps=context.total_steps
                )
            case _:
                raise ValueError(f"Unsupported LR scheduler config ({self._config}).")
