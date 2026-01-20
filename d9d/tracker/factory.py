import dataclasses
from typing import Annotated

from pydantic import Field

from .base import BaseTracker
from .provider.aim.config import AimConfig
from .provider.null import NullTracker, NullTrackerConfig

AnyTrackerConfig = Annotated[AimConfig | NullTrackerConfig, Field(discriminator="provider")]


@dataclasses.dataclass
class _TrackerImportFailed:
    dependency: str
    exception: ImportError


_MAP: dict[type[AnyTrackerConfig], type[BaseTracker] | _TrackerImportFailed] = {
    NullTrackerConfig: NullTracker
}

try:
    from .provider.aim.tracker import AimTracker

    _MAP[AimConfig] = AimTracker
except ImportError as e:
    _MAP[AimConfig] = _TrackerImportFailed(dependency="aim", exception=e)


def tracker_from_config(config: AnyTrackerConfig) -> BaseTracker:
    """
    Instantiates a specific tracker implementation based on the configuration.

    Based on the 'provider' field in the config, this function selects the
    appropriate backend (e.g., Aim, Null). It handles checking for missing
    dependencies for optional backends.

    Args:
        config: A specific tracker configuration object.

    Returns:
        An initialized BaseTracker instance.

    Raises:
        ImportError: If the dependencies for the requested provider are not installed.
    """

    tracker_type = _MAP[type(config)]

    if isinstance(tracker_type, _TrackerImportFailed):
        raise ImportError(  # noqa: TRY004
            f"The tracker configuration {config.provider} could not be loaded - "
            f"ensure these dependencies are installed: {tracker_type.dependency}"
        ) from tracker_type.exception

    return tracker_type.from_config(config)
