from re import Pattern
from typing import Literal

from pydantic import BaseModel, ConfigDict


class FullTuneConfig(BaseModel):
    """Configuration for full fine-tuning of the modules that match a regular expression.

    Attributes:
        kind: Discriminator field. Always ``"full_tune"``.
        module_name_pattern: Regular expression that must fully match the names of modules to unfreeze.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["full_tune"] = "full_tune"

    module_name_pattern: Pattern
