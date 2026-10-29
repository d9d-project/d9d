import dataclasses

from d9d.loop.event import Event


@dataclasses.dataclass()
class EventTestContext:
    step: int


EVENT_A: Event[EventTestContext] = Event("test.a")
EVENT_B: Event[EventTestContext] = Event("test.b")
