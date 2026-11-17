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
    try:
        tracker_type = _tracker_type(config)
    except ImportError as e:
        raise ImportError(
            f"Tracker provider ({config.provider}) cannot be loaded. Install the d9d[tracker-{config.provider}] extra."
        ) from e
    return tracker_type.from_config(config)


def _tracker_type(config: AnyTrackerConfig) -> type[BaseTracker]:
    # Each SDK takes seconds to import, so a provider imports it only when a job selects it.
    match config:
        case LogTrackerConfig():
            return LogTracker
        case NullTrackerConfig():
            return NullTracker
        case AimConfig():
            from .provider.aim.tracker import AimTracker  # ruff: ignore[import-outside-top-level]

            return AimTracker
        case WandbConfig():
            from .provider.wandb.tracker import WandbTracker  # ruff: ignore[import-outside-top-level]

            return WandbTracker
        case TensorBoardConfig():
            from .provider.tensorboard.tracker import TensorBoardTracker  # ruff: ignore[import-outside-top-level]

            return TensorBoardTracker
        case MLflowConfig():
            from .provider.mlflow.tracker import MLflowTracker  # ruff: ignore[import-outside-top-level]

            return MLflowTracker
        case _:
            raise ValueError(f"Tracker configuration type ({type(config).__name__}) is not supported.")
