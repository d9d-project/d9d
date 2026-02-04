from typing import Any

from torch.distributed.checkpoint.stateful import Stateful

from d9d.loop.config import StepActionPeriod, StepActionSpecial


class Stepper(Stateful):
    def __init__(self, initial_step: int, total_steps: int):
        self._current_step = initial_step
        self._total_steps = total_steps

    def step(self):
        self._current_step += 1

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_step": self._current_step,
            "total_steps": self._total_steps
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["total_steps"] != self._total_steps:
            raise ValueError(f'Step count differs: saved {state_dict["total_steps"]}, '
                             f'current {self._total_steps}. Perhaps project configuration changed?')

        self._current_step = state_dict["current_step"]

    def should_do_action(self, action: StepActionPeriod, enable_on_last_step_if_periodic: bool = False) -> bool:
        match action:
            case StepActionSpecial.disable:
                return False
            case StepActionSpecial.last_step:
                return self._current_step == self._total_steps
            case int():
                if action <= 0:
                    raise ValueError()

                will_do_periodic = self._current_step % action == 0
                will_do_last = enable_on_last_step_if_periodic and self._current_step == self._total_steps

                return will_do_periodic or will_do_last
            case _:
                raise ValueError("Invalid step configuration")
