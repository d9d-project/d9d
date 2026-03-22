import dataclasses

import pytest
from d9d.loop.event import Event, EventBus

from d9d_test.loop.event.common_events import EVENT_A, EVENT_B, EventTestContext


@pytest.mark.local
def test_event_creation():
    event = Event(id="test.event")

    assert event.id == "test.event"


@pytest.mark.local
def test_event_is_frozen():
    event = Event(id="test.event")

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.id = "changed"


@pytest.mark.local
def test_event_equality():
    a = Event(id="x")
    b = Event(id="x")
    c = Event(id="y")

    assert a == b
    assert a != c


@pytest.mark.local
def test_event_hashable():
    a = Event(id="x")
    b = Event(id="x")

    assert hash(a) == hash(b)
    assert {a, b} == {a}


@pytest.mark.local
def test_trigger_no_subscribers():
    bus = EventBus()
    bus.trigger(EVENT_A, EventTestContext(step=0))


@pytest.mark.local
def test_trigger():
    bus = EventBus()
    a_log = []
    b_log = []

    bus.subscribe(EVENT_A, lambda ctx: a_log.append(f"a1-{ctx.step}"))
    bus.subscribe(EVENT_B, lambda ctx: b_log.append(f"b1-{ctx.step}"))
    bus.subscribe(EVENT_A, lambda ctx: a_log.append(f"a2-{ctx.step}"))

    bus.trigger(EVENT_A, EventTestContext(step=0))
    bus.trigger(EVENT_A, EventTestContext(step=1))
    bus.trigger(EVENT_B, EventTestContext(step=2))
    bus.trigger(EVENT_A, EventTestContext(step=3))
    bus.trigger(EVENT_B, EventTestContext(step=4))

    assert a_log == ["a1-0", "a2-0", "a1-1", "a2-1", "a1-3", "a2-3"]
    assert b_log == ["b1-2", "b1-4"]


@pytest.mark.local
def test_exception_propagates_and_stops_later_handlers():
    bus = EventBus()

    log = []

    def boom(_):
        raise RuntimeError("boom")

    bus.subscribe(EVENT_A, lambda ctx: log.append(ctx.step))
    bus.subscribe(EVENT_A, boom)
    bus.subscribe(EVENT_A, lambda ctx: log.append(ctx.step))

    with pytest.raises(RuntimeError, match="boom"):
        bus.trigger(EVENT_A, EventTestContext(step=0))

    assert log == [0]


@pytest.mark.local
def test_bounded_normal_execution() -> None:
    bus = EventBus()
    log = []

    bus.subscribe(EVENT_A, lambda ctx: log.append(f"pre-{ctx.step}"))
    bus.subscribe(EVENT_B, lambda ctx: log.append(f"post-{ctx.step}"))

    with bus.bounded(EVENT_A, EVENT_B, EventTestContext(step=10)):
        log.append("inside")

    assert log == ["pre-10", "inside", "post-10"]


@pytest.mark.local
def test_bounded_exception_execution() -> None:
    bus = EventBus()
    log = []

    bus.subscribe(EVENT_A, lambda ctx: log.append(f"pre-{ctx.step}"))
    bus.subscribe(EVENT_B, lambda ctx: log.append(f"post-{ctx.step}"))

    with pytest.raises(ValueError, match="boom"), bus.bounded(EVENT_A, EVENT_B, EventTestContext(step=10)):
        log.append("inside")
        raise ValueError("boom")

    # The exception aborts the context manager, so EVENT_B should not trigger.
    assert log == ["pre-10", "inside"]
