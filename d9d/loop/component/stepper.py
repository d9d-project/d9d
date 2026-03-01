from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

from d9d.loop.config import StepActionPeriod, StepActionSpecial


class Stepper(Stateful):
    """Manages the current step and total steps for a loop.

    This class implements the `Stateful` protocol to allow saving and
    loading its state during checkpointing operations.
    """

    def __init__(self, initial_step: int, total_steps: int):
        """Constructs a Stepper object.

        Args:
            initial_step: The starting step number.
            total_steps: The total number of steps in the training loop.
        """

        self._current_step = initial_step
        self._total_steps = total_steps

    def step(self):
        """Increments the current step counter by one."""

        self._current_step += 1

    @property
    def current_step(self) -> int:
        """The current step number."""

        return self._current_step

    @property
    def total_steps(self) -> int:
        """The total number of steps configured for the loop."""

        return self._total_steps

    def state_dict(self) -> dict[str, Any]:
        """Retrieves the current state of the stepper.

        Returns:
            A dictionary containing the current step and total steps.
        """

        return {"current_step": self._current_step, "total_steps": self._total_steps}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the stepper state from a state dictionary.

        Args:
            state_dict: The state dictionary to load from.

        Raises:
            ValueError: If the total steps in the state dictionary do not match
                the total steps configured in this stepper.
        """

        if state_dict["total_steps"] != self._total_steps:
            raise ValueError(
                f"Step count differs: saved {state_dict['total_steps']}, "
                f"current {self._total_steps}. Perhaps project configuration changed?"
            )

        self._current_step = state_dict["current_step"]

    def should_do_action(
        self, action: StepActionPeriod, enable_on_last_step_if_periodic: bool = False, is_post_step_action: bool = False
    ) -> bool:
        """Determines whether a specific periodic action should be executed.

        Args:
            action: The configuration defining when the action should occur.
                Can be a special action type or an integer representing the period.
            enable_on_last_step_if_periodic: Whether the action should also be
                forced on the very last step if the action is periodic.
            is_post_step_action: Whether the check is being performed after the
                step has logically incremented. Adjusts the step position to compute
                the period accurately.

        Returns:
            True if the action should be executed at the current point, False otherwise.

        Raises:
            ValueError: If the action period is less than or equal to zero, or
                if the provided action configuration is completely invalid.
        """

        position_shift = 0 if is_post_step_action else 1

        shifted_step = self._current_step + position_shift

        match action:
            case StepActionSpecial.disable:
                return False
            case StepActionSpecial.last_step:
                return shifted_step == self._total_steps
            case int():
                if action <= 0:
                    raise ValueError()

                will_do_periodic = shifted_step % action == 0
                will_do_last = enable_on_last_step_if_periodic and shifted_step == self._total_steps

                return will_do_periodic or will_do_last
            case _:
                raise ValueError("Invalid step configuration")
