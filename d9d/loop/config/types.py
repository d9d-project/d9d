from enum import StrEnum


class StepActionSpecial(StrEnum):
    """Special flag values for configuring periodic actions.

    Attributes:
        last_step: The action runs once, at the last step of the run.
        disable: The action never runs.
    """

    last_step = "last_step"
    disable = "disable"


StepActionPeriod = int | StepActionSpecial
"""Period of a periodic action.

An ``int`` is the period in steps. A ``StepActionSpecial`` runs the action only at the last step or disables it.
"""
