import contextlib
import dataclasses
from collections import defaultdict
from collections.abc import Callable, Generator
from typing import Any, Generic, TypeVar

TContext = TypeVar("TContext")


@dataclasses.dataclass(frozen=True)
class Event(Generic[TContext]):
    """
    Typed event descriptor.

    Each event has a unique string identifier and an associated context type.
    Events are defined as module-level constants and used as keys for
    subscription and triggering.

    Attributes:
        id: Unique identifier for the event.
    """

    id: str


class EventBus:
    """
    A centralized event bus for subscribing to and triggering typed events.

    This class maintains a registry of event handlers and dispatches the
    appropriate context to all registered callbacks when an event is triggered.
    """

    def __init__(self) -> None:
        """
        Constructs an EventBus object.
        """

        self._handlers: dict[Event, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event: Event[TContext], handler: Callable[[TContext], None]) -> None:
        """
        Registers a handler function to be executed when a specific event occurs.

        Args:
            event: The event descriptor to subscribe to.
            handler: The callback function to execute when the event is triggered.
        """

        self._handlers[event].append(handler)

    def trigger(self, event: Event[TContext], context: TContext) -> None:
        """
        Dispatches an event to all its registered handlers with the given context.

        Args:
            event: The event descriptor to trigger.
            context: The data associated with the event to pass to the handlers.
        """

        for handler in self._handlers[event]:
            handler(context)

    @contextlib.contextmanager
    def bounded(
        self, event_pre: Event[TContext], event_post: Event[TContext], context: TContext
    ) -> Generator[None, None, None]:
        """
        Context manager that triggers a pre-event on entry and a post-event on exit.

        Args:
            event_pre: The event to trigger immediately before yielding.
            event_post: The event to trigger immediately after the block completes successfully.
            context: The context object passed to both events.

        Yields:
            None
        """

        self.trigger(event_pre, context)
        yield
        self.trigger(event_post, context)
