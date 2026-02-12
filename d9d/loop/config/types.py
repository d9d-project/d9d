from enum import StrEnum


class StepActionSpecial(StrEnum):
    """
    Special flag values for configuring periodic actions.

    Attributes:
        last_step: Indicates the action should occur exactly once at the
            very end of the training run.
        disable: Indicates the action should never occur.
    """

    last_step = "last_step"
    disable = "disable"


StepActionPeriod = int | StepActionSpecial
"""
Union type representing a configuration for periodic events.

Values:
    int: The period in steps (frequency) at which the event occurs.
    StepActionSpecial: A special flag indicating end-of-run execution or disabling.
"""
