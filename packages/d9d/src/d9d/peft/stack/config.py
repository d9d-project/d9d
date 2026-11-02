from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..full_tune.config import FullTuneConfig
from ..lora.config import LoRAConfig


class PeftStackConfig(BaseModel):
    """Configuration for applying a stack of multiple PEFT methods sequentially.

    Attributes:
        kind: Discriminator field. Always ``"stack"``.
        methods: The PEFT configurations (e.g. LoRA, full tune) to apply in order.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["stack"] = "stack"

    methods: list["AnyPeftConfig"]


# The union lives next to the stack config because they refer to each other. Pydantic resolves the forward
# reference above from this module.
AnyPeftConfig = Annotated[
    LoRAConfig | FullTuneConfig | PeftStackConfig,
    Field(discriminator="kind"),
]
"""Union of all PEFT configurations, discriminated by the ``kind`` field."""
