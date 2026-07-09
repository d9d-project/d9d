from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.protocol import MicrobatchPackStream
from d9d.loop.config import JobScheduleConfig, StepActionPeriod, StepActionSpecial


def _resolve_total_steps(config: JobScheduleConfig, stream: MicrobatchPackStream) -> int:
    data_steps = stream.total_steps

    config_steps = config.total_steps

    if config_steps is not None:
        if data_steps is not None and config_steps > data_steps:
            raise ValueError(
                f"Configured total_steps ({config_steps}) exceeds the number of steps "
                f"available from the data ({data_steps})."
            )
        return config_steps

    if data_steps is not None:
        return data_steps

    raise ValueError(
        "Cannot resolve total_steps: the schedule config does not specify total_steps and "
        "the data stream does not report its length. Please set `total_steps` in the schedule config."
    )


class JobSchedule(Stateful):
    """Tracks the progress and resolves the duration of a job loop."""

    def __init__(self, config: JobScheduleConfig, stream: MicrobatchPackStream):
        """Constructs a JobSchedule object.

        Args:
            config: The schedule configuration carrying the optional explicit step budget.
            stream: The microbatch pack stream driving the loop, consulted for its ``total_steps``.

        Raises:
            ValueError: If "total_steps" cannot be resolved from the config and the stream.
        """
        self._current_step = 0
        self._total_steps = _resolve_total_steps(config, stream)

    def step(self):
        """Increments the current step counter by one."""
        self._current_step += 1

    @property
    def current_step(self) -> int:
        """The current step number."""
        return self._current_step

    @property
    def total_steps(self) -> int:
        """The total number of steps resolved for the loop."""
        return self._total_steps

    def state_dict(self) -> dict[str, Any]:
        """Retrieves the current progress of the schedule.

        Returns:
            A dictionary containing the current step.
        """
        return {"current_step": self._current_step}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the schedule progress from a state dictionary.

        Only the current step is restored; "total_steps" is always taken from the resolution at
        construction time, so the configured budget may change across resumes.

        Args:
            state_dict: The state dictionary to load from.
        """
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
