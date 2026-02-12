import pytest
from d9d.loop.component.stepper import Stepper
from d9d.loop.config import StepActionSpecial


@pytest.mark.local
def test_stepper_initialization_and_stepping():
    stepper = Stepper(initial_step=0, total_steps=10)

    assert stepper.current_step == 0
    assert stepper.total_steps == 10

    stepper.step()
    assert stepper.current_step == 1

    stepper.step()
    assert stepper.current_step == 2

    assert stepper.total_steps == 10


@pytest.mark.local
def test_stepper_state_dict_roundtrip():
    stepper = Stepper(initial_step=5, total_steps=20)
    state = stepper.state_dict()

    new_stepper = Stepper(initial_step=0, total_steps=20)
    new_stepper.load_state_dict(state)

    assert new_stepper.current_step == 5
    new_stepper.step()
    assert new_stepper.current_step == 6


@pytest.mark.local
def test_stepper_load_state_dict_config_mismatch():
    saver = Stepper(initial_step=10, total_steps=100)
    state = saver.state_dict()

    loader = Stepper(initial_step=0, total_steps=200)
    with pytest.raises(ValueError, match="Step count differs"):
        loader.load_state_dict(state)


@pytest.mark.local
@pytest.mark.parametrize(
    ("current_step", "total_steps", "action", "enable_on_last", "expected"),
    [
        # Special: Disable
        (1, 10, StepActionSpecial.disable, False, False),
        (10, 10, StepActionSpecial.disable, False, False),
        # Special: Last Step
        (9, 10, StepActionSpecial.last_step, False, False),
        (10, 10, StepActionSpecial.last_step, False, True),
        # Periodic (every 5 steps) - standard cases
        (4, 20, 5, False, False),
        (5, 20, 5, False, True),
        (6, 20, 5, False, False),
        (10, 20, 5, False, True),
        # Periodic - Boundary check (last step is also periodic)
        (20, 20, 5, False, True),
        # Periodic with enable_on_last_step_if_periodic=True
        # Case 1: Last step is NOT divisible by period (e.g., total=21, period=5)
        (21, 21, 5, True, True),
        # Case 2: Not last step, not divisible
        (20, 21, 5, True, True),  # Wait, 20 % 5 == 0, so True regardless of flag
        (19, 21, 5, True, False),
        # Case 3: Last step IS divisible by period
        (20, 20, 5, True, True),
    ],
)
def test_should_do_action(current_step, total_steps, action, enable_on_last, expected):
    stepper = Stepper(initial_step=0, total_steps=total_steps)
    # Manually set private attribute to simulate state for test
    stepper._current_step = current_step

    assert stepper.should_do_action(action, enable_on_last_step_if_periodic=enable_on_last) is expected


@pytest.mark.local
def test_should_do_action_invalid_input():
    stepper = Stepper(initial_step=1, total_steps=10)

    with pytest.raises(ValueError):
        stepper.should_do_action(0)

    with pytest.raises(ValueError):
        stepper.should_do_action(-5)

    with pytest.raises(ValueError, match="Invalid step configuration"):
        stepper.should_do_action("invalid_action")
