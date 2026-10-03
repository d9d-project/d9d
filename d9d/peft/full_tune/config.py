from re import Pattern
from typing import Literal

from pydantic import BaseModel


class FullTuneConfig(BaseModel):
    """Configuration for full fine-tuning of the modules that match a regular expression.

    Attributes:
        kind: Discriminator field, always ``"full_tune"``.
        module_name_pattern: Regular expression that must fully match the names of modules to unfreeze.
    """

    kind: Literal["full_tune"] = "full_tune"

    module_name_pattern: Pattern
