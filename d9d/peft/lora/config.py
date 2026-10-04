from re import Pattern
from typing import Literal

from pydantic import BaseModel, ConfigDict


class LoRAParameters(BaseModel):
    """Configuration for LoRA layers.

    Attributes:
        r: Rank of the low-rank adaptation matrices.
        alpha: Scaling factor for the learned weights. The LoRA output is scaled by ``alpha / r``.
        dropout: Dropout probability for the input to LoRA layers.
    """

    model_config = ConfigDict(extra="forbid")

    r: int
    alpha: int
    dropout: float


class LoRAConfig(BaseModel):
    """Configuration for LoRA application.

    Attributes:
        kind: Discriminator field. Always ``"lora"``.
        module_name_pattern: Regular expression that must fully match the names of modules to wrap with LoRA.
        params: Hyperparameters for the LoRA layers.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["lora"] = "lora"

    module_name_pattern: Pattern
    params: LoRAParameters
