from re import Pattern
from typing import Literal

from pydantic import BaseModel


class LoRAParameters(BaseModel):
    """
    Hyperparameters for LoRA layers.

    Attributes:
        r: Rank of the low-rank adaptation matrices.
        alpha: Scaling factor for the learned weights.
        dropout: Dropout probability for the input to LoRA layers.
    """

    r: int
    alpha: int
    dropout: float


class LoRAConfig(BaseModel):
    """
    Configuration for LoRA application.

    Attributes:
        kind: Discriminator field, always "lora".
        module_name_pattern: Regular expression matching module names to wrap with LoRA.
        params: Hyperparameters for the LoRA layers.
    """

    kind: Literal["lora"] = "lora"

    module_name_pattern: Pattern
    params: LoRAParameters
