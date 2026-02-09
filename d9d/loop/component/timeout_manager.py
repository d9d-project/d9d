from enum import StrEnum

from d9d.core.dist_context import DistributedContext
from d9d.loop.config.config import TimeoutConfig


class TimeoutState(StrEnum):
    """
    Represents the lifecycle states of the timeout manager configuration.
    """
    none = "none"
    set_initial = "set_initial"
    set_regular = "set_regular"


class TimeoutManager:
    """
    Manages the dynamic adjustment of distributed timeouts during the job loop.

    This manager handles the transition from initialization timeouts (which may need
    to be longer due to JIT compilation, caching, or startup overhead) to regular
    step execution timeouts.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            config: TimeoutConfig
    ):
        """
        Constructs the TimeoutManager object.

        Args:
            dist_context: The distributed context where timeouts are applied.
            config: Configuration containing initialization and step timeout values.
        """

        self._dist_context = dist_context
        self._config = config
        self._state = TimeoutState.none

    def set_init(self):
        """
        Sets the distributed backend timeout to the initialization value.

        This allows for a longer timeout duration during the startup phase of the
        application where compilation or heavy loading operations might occur.

        Raises:
            ValueError: If the timeout state has already been initialized.
        """

        if self._state != TimeoutState.none:
            raise ValueError("Can only set init timeout from initial state")

        self._dist_context.set_timeout(self._config.init_timeout)
        self._state = TimeoutState.set_initial

    def set_periodic(self):
        """
        Transitions the distributed backend timeout to the regular step value.

        If the manager is currently in the initialization state, this updates the
        backend timeout to the configured step timeout. If already in the regular
        state, no action is taken.

        Raises:
            ValueError: If the manager has not been initialized.
        """
        match self._state:
            case TimeoutState.set_initial:
                self._dist_context.set_timeout(self._config.step_timeout)
                self._state = TimeoutState.set_regular
            case TimeoutState.set_regular:
                pass  # do nothing
            case _:
                raise ValueError("Unknown timeout state")
