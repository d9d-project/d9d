import dataclasses

from .curves import CurveBase


@dataclasses.dataclass
class SchedulePhase:
    """A single phase of a piecewise schedule.

    Attributes:
        start_step: Global step at which this phase starts.
        end_step: Global step at which this phase ends (exclusive).
        start_value: Multiplier at ``start_step``.
        end_value: Multiplier at ``end_step``.
        curve: Curve that interpolates between ``start_value`` and ``end_value``.
    """

    start_step: int
    end_step: int
    start_value: float
    end_value: float
    curve: CurveBase


class PiecewiseScheduleEngine:
    """Engine that computes learning rate multipliers from a list of phases."""

    def __init__(self, phases: list[SchedulePhase]):
        """Constructs the ``PiecewiseScheduleEngine`` object.

        Args:
            phases: Schedule phases, in order.

        Raises:
            ValueError: If ``phases`` is empty.
        """
        if len(phases) == 0:
            raise ValueError("A piecewise schedule must contain at least one phase.")

        self._phases = phases

    def get_factor(self, step: int) -> float:
        """Computes the learning rate multiplier for the given step.

        Args:
            step: The global training step.

        Returns:
            The multiplier. Steps before the first phase get its start value. Steps after the last phase get
            its end value.
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
