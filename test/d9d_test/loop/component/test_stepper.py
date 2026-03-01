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
    stepper = Stepper(initial_step=0, total_steps=total_steps)
    # Manually set private attribute to simulate state for test
    stepper._current_step = current_step

    assert (
        stepper.should_do_action(
            action,
            enable_on_last_step_if_periodic=enable_on_last,
            is_post_step_action=is_post_step,
        )
        is expected
    )


@pytest.mark.local
def test_should_do_action_invalid_input():
    stepper = Stepper(initial_step=1, total_steps=10)

    with pytest.raises(ValueError):
        stepper.should_do_action(0)

    with pytest.raises(ValueError):
        stepper.should_do_action(-5)

    with pytest.raises(ValueError, match="Invalid step configuration"):
        stepper.should_do_action("invalid_action")
