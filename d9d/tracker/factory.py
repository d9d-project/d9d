import dataclasses
from typing import Annotated

from pydantic import Field

from .base import BaseTracker
from .provider.aim.config import AimConfig
from .provider.null import NullTracker, NullTrackerConfig

AnyTrackerConfig = Annotated[AimConfig | NullTrackerConfig, Field(discriminator="provider")]
"""Union of all tracker configurations, discriminated by the ``provider`` field."""


@dataclasses.dataclass
class _TrackerImportFailed:
    dependency: str
    exception: ImportError


_MAP: dict[type[AnyTrackerConfig], type[BaseTracker] | _TrackerImportFailed] = {
    NullTrackerConfig: NullTracker,
}

try:
    from .provider.aim.tracker import AimTracker

    _MAP[AimConfig] = AimTracker
except ImportError as e:
    _MAP[AimConfig] = _TrackerImportFailed(dependency="aim", exception=e)


def tracker_from_config(config: AnyTrackerConfig) -> BaseTracker:
    """Creates the tracker that the ``provider`` field of the configuration selects.

    Args:
        config: The tracker configuration.

    Returns:
        The tracker.

    Raises:
        ImportError: If the optional dependency of the provider is not installed.
    """
    tracker_type = _MAP[type(config)]

    if isinstance(tracker_type, _TrackerImportFailed):
        raise ImportError(
            f"Tracker provider ({config.provider}) cannot be loaded. "
            f"Install its dependency ({tracker_type.dependency})."
        ) from tracker_type.exception

    return tracker_type.from_config(config)
