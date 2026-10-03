from enum import StrEnum

from d9d.core.dist_context import DistributedContext
from d9d.loop.config.config import TimeoutConfig


class TimeoutState(StrEnum):
    """Lifecycle states of the ``TimeoutManager``.

    Attributes:
        none: No timeout was set yet.
        set_initial: The initialization timeout is active.
        set_regular: The step timeout is active.
    """

    none = "none"
    set_initial = "set_initial"
    set_regular = "set_regular"


class TimeoutManager:
    """Manages the dynamic adjustment of distributed timeouts during the job loop.

    The manager switches from the initialization timeout to the step timeout. The initialization
    timeout can be longer to cover compilation, caching and other startup work.
    """

    def __init__(self, dist_context: DistributedContext, config: TimeoutConfig):
        """Constructs the ``TimeoutManager`` object.

        Args:
            dist_context: The distributed context where timeouts are applied.
            config: Configuration containing initialization and step timeout values.
        """
        self._dist_context = dist_context
        self._config = config
        self._state = TimeoutState.none

    def set_init(self):
        """Sets the distributed backend timeout to the initialization value.

        Raises:
            ValueError: If a timeout was already set.
        """
        if self._state != TimeoutState.none:
            raise ValueError(
                f"Timeout state ({self._state}) is not initial. Call set_init() only once, before set_periodic()."
            )

        self._dist_context.set_timeout(self._config.init_timeout)
        self._state = TimeoutState.set_initial

    def set_periodic(self):
        """Transitions the distributed backend timeout to the regular step value.

        Does nothing if the step timeout is already set.

        Raises:
            ValueError: If ``set_init`` was not called before.
        """
        match self._state:
            case TimeoutState.set_initial:
                self._dist_context.set_timeout(self._config.step_timeout)
                self._state = TimeoutState.set_regular
            case TimeoutState.set_regular:
                pass
            case _:
                raise ValueError(
                    f"Timeout state ({self._state}) has no init timeout. Call set_init() before set_periodic()."
                )
