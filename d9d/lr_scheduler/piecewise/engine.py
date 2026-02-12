import dataclasses

from .curves import CurveBase


@dataclasses.dataclass
class SchedulePhase:
    """
    Data container representing a single phase in a piecewise schedule.

    Attributes:
        start_step: The absolute step index where this phase begins.
        end_step: The absolute step index where this phase ends.
        start_value: The multiplier value at start_step.
        end_value: The multiplier value at end_step.
        curve: The interpolation logic for this phase.
    """

    start_step: int
    end_step: int
    start_value: float
    end_value: float
    curve: CurveBase


class PiecewiseScheduleEngine:
    """
    Runtime engine that calculates multipliers based on a list of defined phases.
    """

    def __init__(self, phases: list[SchedulePhase]):
        """
        Constructs the schedule engine.

        Args:
            phases: A sequential list of schedule phases.

        Raises:
            ValueError: If the phases list is empty.
        """

        if len(phases) == 0:
            raise ValueError("Scheduler should contain at least one phase")

        self._phases = phases

    def get_factor(self, step: int) -> float:
        """
        Computes the learning rate multiplier for the given step.

        Args:
            step: The global training step.

        Returns:
            The calculated multiplier. If the step is outside defined phases,
            it clamps to the nearest boundary value.
        """

        if step < 0:
            return self._phases[0].start_value

        for phase in self._phases:
            if not (phase.start_step <= step < phase.end_step):
                continue

            steps_in_phase = step - phase.start_step
            phase_len = phase.end_step - phase.start_step
            phase_progress = steps_in_phase / phase_len

            return phase.curve.compute(start=phase.start_value, end=phase.end_value, step_p=phase_progress)

        return self._phases[-1].end_value
