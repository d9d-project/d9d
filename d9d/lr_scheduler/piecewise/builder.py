from typing import Self

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from d9d.core.protocol import LRSchedulerProtocol

from .curves import CurveBase
from .engine import PiecewiseScheduleEngine, SchedulePhase


class PiecewiseScheduleBuilder:
    """
    Builder for constructing multiphase learning rate schedules.
    """

    def __init__(
            self,
            initial_multiplier: float,
            total_steps: int | None
    ):
        """
        Constructs a new PiecewiseScheduleBuilder.

        Args:
            initial_multiplier: The starting learning rate multiplier (usually 0.0 or 1.0).
            total_steps: The total number of training steps. Required if using percentage-based methods.
        """

        self._phases: list[SchedulePhase] = []
        self._total_steps = total_steps
        self._last_end_step = 0
        self._last_multiplier = initial_multiplier

    def for_steps(self, steps: int, target_multiplier: float, curve: CurveBase) -> Self:
        """
        Adds a schedule phase lasting for a specific number of steps.

        Args:
            steps: Duration of this phase in steps.
            target_multiplier: The value of the multiplier at the end of this phase.
            curve: The interpolation curve to use for bridging the start and end values.

        Returns:
            The builder instance for chaining.
        """

        self._phases.append(SchedulePhase(
            start_step=self._last_end_step,
            end_step=self._last_end_step + steps,
            curve=curve,
            start_value=self._last_multiplier,
            end_value=target_multiplier
        ))

        self._last_end_step += steps
        self._last_multiplier = target_multiplier

        return self

    def until_percentage(self, p: float, target_multiplier: float, curve: CurveBase) -> Self:
        """
        Adds a schedule phase lasting until a specific percentage of total training steps is reached.

        Args:
            p: The target percentage (0.0 to 1.0) of total_steps where this phase ends.
            target_multiplier: The value of the multiplier at the end of this phase.
            curve: The interpolation curve to use.

        Returns:
            The builder instance for chaining.

        Raises:
            ValueError: If total_steps was not provided in constructor or if the target
                percentage implies a step count earlier than the current cursor.
        """

        if self._total_steps is None:
            raise ValueError(
                "You must define 'total_steps' in the constructor to use percentage-based methods."
            )

        if not 0.0 <= p <= 1.0:
            raise ValueError("Percentage should be in range of [0.0, 1.0]")

        target_step_abs = int(self._total_steps * p)
        duration = target_step_abs - self._last_end_step

        if duration < 0:
            raise ValueError(
                f"Target percentage {p} (step {target_step_abs}) is behind "
                f"current cursor (step {self._last_end_step})."
            )

        return self.for_steps(duration, target_multiplier, curve)

    def fill_rest(self, target_multiplier: float, curve: CurveBase) -> Self:
        """
        Adds a schedule phase that lasts from the current cursor until the end of training.

        Args:
            target_multiplier: The value of the multiplier at the very end of training.
            curve: The interpolation curve to use.

        Returns:
            The builder instance for chaining.
        """

        return self.until_percentage(1.0, target_multiplier, curve)

    def build(self, optimizer: Optimizer) -> LRSchedulerProtocol:
        """
        Finalizes the schedule and returns a PyTorch LR Scheduler.

        Args:
            optimizer: The optimizer to wrap.

        Returns:
            A scheduler configured with the defined phases.

        Raises:
            ValueError: If the defined phases exceed the total_steps provided.
        """

        if self._total_steps is not None and self._last_end_step > self._total_steps:
            raise ValueError(
                f"Schedule defined for {self._last_end_step} steps, but total_steps is {self._total_steps}."
            )

        engine = PiecewiseScheduleEngine(self._phases)
        return LambdaLR(optimizer, engine.get_factor)


def piecewise_schedule(
        initial_multiplier: float,
        total_steps: int | None = None
) -> PiecewiseScheduleBuilder:
    """
    Entry point for creating a piecewise learning rate schedule.

    Args:
        initial_multiplier: The initial learning rate multiplier.
        total_steps: Total training steps. Required for percentage-based scheduling.

    Returns:
        A builder instance to configure phases.
    """

    return PiecewiseScheduleBuilder(
        initial_multiplier=initial_multiplier,
        total_steps=total_steps
    )
