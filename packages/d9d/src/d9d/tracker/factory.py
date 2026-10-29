from typing import Annotated

from pydantic import Field

from .base import BaseTracker
from .provider.aim.config import AimConfig
from .provider.log import LogTracker, LogTrackerConfig
from .provider.mlflow.config import MLflowConfig
from .provider.null import NullTracker, NullTrackerConfig
from .provider.tensorboard.config import TensorBoardConfig
from .provider.wandb.config import WandbConfig

AnyTrackerConfig = Annotated[
    AimConfig | LogTrackerConfig | MLflowConfig | NullTrackerConfig | TensorBoardConfig | WandbConfig,
    Field(discriminator="provider"),
]
"""Union of all tracker configurations, discriminated by the ``provider`` field."""


def tracker_from_config(config: AnyTrackerConfig) -> BaseTracker:
    """Creates the tracker that the ``provider`` field of the configuration selects.

    Args:
        config: The tracker configuration.

    Returns:
        The tracker.

    Raises:
        ImportError: If the optional dependency of the provider is not installed.
        ValueError: If the configuration type is unknown.
    """
    # Each SDK takes seconds to import, so a provider imports it only when a job selects it.
    try:
        match config:
            case LogTrackerConfig():
                return LogTracker.from_config(config)
            case NullTrackerConfig():
                return NullTracker.from_config(config)
            case AimConfig():
                from .provider.aim.tracker import AimTracker  # noqa: PLC0415

                return AimTracker.from_config(config)
            case WandbConfig():
                from .provider.wandb.tracker import WandbTracker  # noqa: PLC0415

                return WandbTracker.from_config(config)
            case TensorBoardConfig():
                from .provider.tensorboard.tracker import TensorBoardTracker  # noqa: PLC0415

                return TensorBoardTracker.from_config(config)
            case MLflowConfig():
                from .provider.mlflow.tracker import MLflowTracker  # noqa: PLC0415

                return MLflowTracker.from_config(config)
            case _:
                raise ValueError(f"Tracker configuration type ({type(config).__name__}) is not supported.")
    except ImportError as e:
        raise ImportError(
            f"Tracker provider ({config.provider}) cannot be loaded. Install the d9d[tracker-{config.provider}] extra."
        ) from e
