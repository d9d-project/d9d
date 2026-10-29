import pytest
import torch
from d9d.loop.component import JobSchedule, TrackerWriter
from d9d.loop.component.tracker_writer import PendingTrackerWrites
from d9d.loop.config import JobScheduleConfig
from d9d.tracker import BaseTrackerRun, Histogram, Scalar


class _FakeStream:
    @property
    def total_steps(self) -> int | None:
        return None


class _RecordingRun(BaseTrackerRun):
    def __init__(self):
        self.writes = []

    def write(self, point, value):
        match value:
            case Scalar():
                self.writes.append(("scalar", point.name, float(value.value), point.step, point.context))
            case Histogram():
                self.writes.append(("histogram", point.name, value.counts.tolist(), point.step, point.context))


@pytest.mark.local
@pytest.mark.parametrize(
    ("max_pending_steps", "must_be_written_before_exit"),
    [
        # The first write of step 6 waits for the value of step 5.
        (1, 1),
        (2, 0),
    ],
)
def test_writer(max_pending_steps, must_be_written_before_exit):
    run = _RecordingRun()
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=100), stream=_FakeStream())
    schedule._current_step = 5

    with PendingTrackerWrites(run, max_pending_steps=max_pending_steps) as pending:
        writer = TrackerWriter(pending, schedule=schedule, context={"stage": "train"})

        # fp64 needs no conversion, so the copy reads the tensor itself.
        loss = torch.tensor(1.5, device="cuda", dtype=torch.float64)
        writer.log("loss", Scalar(value=loss))
        # The logged value is read at the call, so a later in-place change on the same stream does not affect it.
        loss.add_(1.0)
        schedule.step()
        writer.log("lr", Scalar(value=1e-5), context={"group": "0"})
        writer.log("tokens", Scalar(value=torch.tensor(1_000_000_003, device="cuda")))
        writer.log("counts", Histogram(counts=torch.arange(4.0, device="cuda")))

        with pytest.raises(ValueError, match="one element"):
            Scalar(value=torch.zeros(2))
        with pytest.raises(ValueError, match="one-dimensional"):
            Histogram(counts=torch.zeros(2, 2))

        assert len(run.writes) >= must_be_written_before_exit

    assert run.writes == [
        ("scalar", "loss", 1.5, 5, {"stage": "train"}),
        ("scalar", "lr", 1e-5, 6, {"stage": "train", "group": "0"}),
        ("scalar", "tokens", 1_000_000_003, 6, {"stage": "train"}),
        ("histogram", "counts", [0.0, 1.0, 2.0, 3.0], 6, {"stage": "train"}),
    ]

    with pytest.raises(RuntimeError, match="closed"):
        writer.log("loss", Scalar(value=1.0))
