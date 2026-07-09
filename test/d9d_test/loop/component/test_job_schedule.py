import pytest
from d9d.loop.component.job_schedule import JobSchedule
from d9d.loop.config import JobScheduleConfig, StepActionSpecial


class FakeStream:
    def __init__(self, total_steps: int | None):
        self._total_steps = total_steps

    @property
    def total_steps(self) -> int | None:
        return self._total_steps


@pytest.mark.local
def test_job_schedule_initialization_and_stepping():
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=10), stream=FakeStream(None))

    assert schedule.current_step == 0
    assert schedule.total_steps == 10

    schedule.step()
    assert schedule.current_step == 1

    schedule.step()
    assert schedule.current_step == 2

    assert schedule.total_steps == 10


@pytest.mark.local
@pytest.mark.parametrize(
    ("config_total_steps", "stream", "expected"),
    [
        (10, FakeStream(None), 10),  # config set, stream length unknown -> config
        (10, FakeStream(20), 10),  # config set, length >= config -> config
        (20, FakeStream(20), 20),  # config set, length == config -> config
        (None, FakeStream(20), 20),  # config unset, stream length known -> length
    ],
)
def test_job_schedule_total_steps_resolution(config_total_steps, stream, expected):
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=config_total_steps), stream=stream)
    assert schedule.total_steps == expected


@pytest.mark.local
def test_job_schedule_total_steps_config_exceeds_data():
    with pytest.raises(ValueError, match="exceeds the number of steps"):
        JobSchedule(config=JobScheduleConfig(total_steps=30), stream=FakeStream(20))


@pytest.mark.local
def test_job_schedule_total_steps_unresolvable():
    with pytest.raises(ValueError, match="Cannot resolve total_steps"):
        JobSchedule(config=JobScheduleConfig(total_steps=None), stream=FakeStream(None))


@pytest.mark.local
def test_job_schedule_state_dict_roundtrip():
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=20), stream=FakeStream(None))
    schedule.step()
    schedule.step()
    schedule.step()
    schedule.step()
    schedule.step()
    state = schedule.state_dict()

    new_schedule = JobSchedule(config=JobScheduleConfig(total_steps=20), stream=FakeStream(None))
    new_schedule.load_state_dict(state)

    assert new_schedule.current_step == 5
    new_schedule.step()
    assert new_schedule.current_step == 6


@pytest.mark.local
def test_job_schedule_load_state_dict_allows_changed_budget():
    saver = JobSchedule(config=JobScheduleConfig(total_steps=100), stream=FakeStream(None))
    state = saver.state_dict()

    # The budget can change across resumes - only the current step is restored.
    loader = JobSchedule(config=JobScheduleConfig(total_steps=200), stream=FakeStream(None))
    loader.load_state_dict(state)

    assert loader.current_step == 0
    assert loader.total_steps == 200


@pytest.mark.local
@pytest.mark.parametrize(
    ("current_step", "total_steps", "action", "enable_on_last", "is_post_step", "expected"),
    [
        # --- Pre-step actions (is_post_step=False) ---
        # Special: Disable
        (1, 10, StepActionSpecial.disable, False, False, False),
        (9, 10, StepActionSpecial.disable, False, False, False),
        # Special: Last Step - triggers when current_step == total_steps - 1
        (8, 10, StepActionSpecial.last_step, False, False, False),
        (9, 10, StepActionSpecial.last_step, False, False, True),
        # Periodic (every 5 steps) - triggers when (current_step + 1) % 5 == 0, i.e. 4, 9, 14, 19...
        (3, 20, 5, False, False, False),
        (4, 20, 5, False, False, True),
        (5, 20, 5, False, False, False),
        (9, 20, 5, False, False, True),
        # Periodic - Boundary check (last step is also periodic: total=20, period=5, last step at 19)
        (19, 20, 5, False, False, True),
        # Periodic with enable_on_last_step_if_periodic=True
        (20, 21, 5, True, False, True),  # not periodic, but is last → True via flag
        (19, 21, 5, True, False, True),  # (19+1)%5=0 → True regardless of flag
        (18, 21, 5, True, False, False),  # neither periodic nor last
        (19, 20, 5, True, False, True),  # Last step IS also on a periodic tick
        # --- Post-step actions (is_post_step=True) ---
        # Special: Disable
        (1, 10, StepActionSpecial.disable, False, True, False),
        (10, 10, StepActionSpecial.disable, False, True, False),
        # Special: Last Step - triggers when current_step == total_steps
        (9, 10, StepActionSpecial.last_step, False, True, False),
        (10, 10, StepActionSpecial.last_step, False, True, True),
        # Periodic (every 5 steps) - triggers when current_step % 5 == 0, i.e. 5, 10, 15, 20...
        (4, 20, 5, False, True, False),
        (5, 20, 5, False, True, True),
        (9, 20, 5, False, True, False),
        (10, 20, 5, False, True, True),
        # Periodic - Boundary check (last step is also periodic: total=20, period=5, last step at 20)
        (20, 20, 5, False, True, True),
        # Periodic with enable_on_last_step_if_periodic=True
        (21, 21, 5, True, True, True),  # not periodic, but is last → True via flag
        (20, 21, 5, True, True, True),  # 20%5=0 → True regardless of flag
        (19, 21, 5, True, True, False),  # neither periodic nor last
        (20, 20, 5, True, True, True),  # Last step IS also on a periodic tick
    ],
)
def test_should_do_action(
    current_step,
    total_steps,
    action,
    enable_on_last,
    is_post_step,
    expected,
) -> None:
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=total_steps), stream=FakeStream(None))
    # Manually set private attribute to simulate state for test
    schedule._current_step = current_step

    assert (
        schedule.should_do_action(
            action,
            enable_on_last_step_if_periodic=enable_on_last,
            is_post_step_action=is_post_step,
        )
        is expected
    )


@pytest.mark.local
def test_should_do_action_invalid_input():
    schedule = JobSchedule(config=JobScheduleConfig(total_steps=10), stream=FakeStream(None))

    with pytest.raises(ValueError):
        schedule.should_do_action(0)

    with pytest.raises(ValueError):
        schedule.should_do_action(-5)

    with pytest.raises(ValueError, match="Invalid step configuration"):
        schedule.should_do_action("invalid_action")
