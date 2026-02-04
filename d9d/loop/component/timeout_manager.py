from enum import StrEnum

from d9d.core.dist_context import DistributedContext
from d9d.loop.config.config import TimeoutConfig


class TimeoutState(StrEnum):
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
        Constructs the timeout manager.

        Args:
            dist_context: The distributed context where timeouts are applied.
            config: Configuration containing initialization and step timeout values.
        """

        self._dist_context = dist_context
        self._config = config
        self._state = TimeoutState.none

    def step(self):
        """
        Updates the distributed backend timeout based on the current phase.
        """

        match self._state:
            case TimeoutState.none:
                self._dist_context.set_timeout(self._config.init_timeout)
                self._state = TimeoutState.set_initial
            case TimeoutState.set_initial:
                self._dist_context.set_timeout(self._config.step_timeout)
                self._state = TimeoutState.set_regular
            case TimeoutState.set_regular:
                pass  # do nothing
            case _:
                raise ValueError("Unknown timeout state")
