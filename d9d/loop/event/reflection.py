import inspect
from collections.abc import Callable

from d9d.loop.event import TContext

from .core import Event, EventBus

_SUBSCRIBE_MARKER = "_d9d_subscribed_events"


def subscribe(event: Event[TContext]) -> Callable[[Callable[[TContext], None]], Callable[[TContext], None]]:
    """
    Decorator that tags a method to be subscribed to specific event.

    This decorator does not register the method immediately. Instead, it attaches
    metadata to the function. To finalize registration, use `subscribe_annotated()`.

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
    """
    Automatically subscribes all methods on the target object decorated with `@subscribe`.

    This method uses introspection to find tagged methods and binds them to the
    provided event bus.

    Args:
        bus: The EventBus instance to register the handlers to.
        target: The initialized class instance containing the decorated methods.
    """
    for _, method in inspect.getmembers(target, predicate=inspect.ismethod):
        event: Event | None = getattr(method.__func__, _SUBSCRIBE_MARKER, None)

        if event is not None:
            bus.subscribe(event, method)
