from enum import StrEnum

import torch


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


class DtypeName(StrEnum):
    """Name of a floating-point ``torch`` dtype in a configuration.

    Attributes:
        float32: ``torch.float32``.
        bfloat16: ``torch.bfloat16``.
        float16: ``torch.float16``.
    """

    float32 = "float32"
    bfloat16 = "bfloat16"
    float16 = "float16"


def dtype_from_name(name: DtypeName) -> torch.dtype:
    """Returns the ``torch`` dtype with the given name.

    Args:
        name: The dtype name.

    Returns:
        The dtype.

    Raises:
        ValueError: If ``name`` is not a ``DtypeName``.
    """
    match name:
        case DtypeName.float32:
            return torch.float32
        case DtypeName.bfloat16:
            return torch.bfloat16
        case DtypeName.float16:
            return torch.float16
        case _:
            raise ValueError(f"Unknown dtype name ({name}).")
