import pytest
from d9d.loop.event import EventBus, subscribe, subscribe_annotated

from d9d_test.loop.event.common_events import EVENT_A, EVENT_B, EventTestContext


@pytest.mark.local
def test_subscribe_annotated_basic():
    class MySubscriber:
        def __init__(self):
            self.log = []

        @subscribe(EVENT_A)
        def handle_a(self, ctx: EventTestContext):
            self.log.append(f"A:{ctx.step}")

        @subscribe(EVENT_B)
        def handle_b(self, ctx: EventTestContext):
            self.log.append(f"B:{ctx.step}")

        def not_subscribed(self, ctx: EventTestContext):
            self.log.append(f"FAIL:{ctx.step}")

    bus = EventBus()
    subscriber = MySubscriber()

    subscribe_annotated(bus, subscriber)

    bus.trigger(EVENT_A, EventTestContext(step=1))
    bus.trigger(EVENT_B, EventTestContext(step=2))
    bus.trigger(EVENT_A, EventTestContext(step=3))

    assert subscriber.log == ["A:1", "B:2", "A:3"]


@pytest.mark.local
def test_subscribe_annotated_multiple_instances():
    class CounterSubscriber:
        def __init__(self, name: str):
            self.name = name
            self.count = 0

        @subscribe(EVENT_A)
        def increment(self, ctx: EventTestContext):
            self.count += ctx.step

    bus = EventBus()
    sub1 = CounterSubscriber("first")
    sub2 = CounterSubscriber("second")

    # Register both instances
    subscribe_annotated(bus, sub1)
    subscribe_annotated(bus, sub2)

    bus.trigger(EVENT_A, EventTestContext(step=5))
    bus.trigger(EVENT_A, EventTestContext(step=10))

    # They should both receive the events without interfering with each other
    assert sub1.count == 15
    assert sub2.count == 15


@pytest.mark.local
def test_subscribe_annotated_multiple_methods_same_event():
    class MultiSubscriber:
        def __init__(self):
            self.log = []

        @subscribe(EVENT_A)
        def first_handler(self, ctx: EventTestContext):
            self.log.append("first")

        @subscribe(EVENT_A)
        def second_handler(self, ctx: EventTestContext):
            self.log.append("second")

    bus = EventBus()
    sub = MultiSubscriber()

    subscribe_annotated(bus, sub)
    bus.trigger(EVENT_A, EventTestContext(step=0))

    # Order of execution for `inspect.getmembers` might be alphabetical,
    # but the key part is that both executed.
    assert len(sub.log) == 2
    assert "first" in sub.log
    assert "second" in sub.log


@pytest.mark.local
def test_subscribe_annotated_with_inheritance():
    class BaseSubscriber:
        def __init__(self):
            self.log = []

        @subscribe(EVENT_A)
        def handle_base(self, ctx: EventTestContext):
            self.log.append(f"base_a:{ctx.step}")

    class ChildSubscriber(BaseSubscriber):
        @subscribe(EVENT_B)
        def handle_child(self, ctx: EventTestContext):
            self.log.append(f"child_b:{ctx.step}")

    bus = EventBus()
    child = ChildSubscriber()

    subscribe_annotated(bus, child)

    bus.trigger(EVENT_A, EventTestContext(step=1))
    bus.trigger(EVENT_B, EventTestContext(step=2))

    # Both inherited base handler and child handler should be registered correctly
    assert child.log == ["base_a:1", "child_b:2"]
