from re import Pattern
from typing import Literal

from pydantic import BaseModel


class FullTuneConfig(BaseModel):
    """
    Configuration for Full Fine-Tuning.

    Allows specifying which modules should be fully fine-tuned using regex patterns.

    Attributes:
        kind: Discriminator field, always "full_tune".
        module_name_pattern: Regular expression matching module names to unfreeze.
    """

    kind: Literal["full_tune"] = "full_tune"

    module_name_pattern: Pattern
