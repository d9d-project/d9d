import inspect
from collections.abc import Callable

from d9d.loop.event import TContext

from .core import Event, EventBus

_SUBSCRIBE_MARKER = "_d9d_subscribed_events"


def subscribe(event: Event[TContext]) -> Callable[[Callable[[TContext], None]], Callable[[TContext], None]]:
    """Marks a method to be subscribed to an event.

    The decorator does not register the method. Call ``subscribe_annotated`` on the instance to register it.

    Args:
        event: Event descriptor to bind this method to.

    Returns:
        The decorated function.
    """

    def decorator(func: Callable) -> Callable:
        setattr(func, _SUBSCRIBE_MARKER, event)
        return func

    return decorator


def subscribe_annotated(bus: EventBus, target: object) -> None:
    """Subscribes all methods of the target object that are decorated with ``@subscribe``.

    Args:
        bus: The event bus to register the handlers on.
        target: The initialized class instance containing the decorated methods.
    """
    for _, method in inspect.getmembers(target, predicate=inspect.ismethod):
        event: Event | None = getattr(method.__func__, _SUBSCRIBE_MARKER, None)

        if event is not None:
            bus.subscribe(event, method)
