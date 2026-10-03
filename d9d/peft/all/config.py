from typing import Annotated, Literal

from pydantic import BaseModel, Field

from d9d.peft.full_tune.config import FullTuneConfig
from d9d.peft.lora.config import LoRAConfig


class PeftStackConfig(BaseModel):
    """Configuration for applying a stack of multiple PEFT methods sequentially.

    Attributes:
        kind: Discriminator field, always ``"stack"``.
        methods: The PEFT configurations (e.g. LoRA, full tune) to apply in order.
    """

    kind: Literal["stack"] = "stack"

    methods: list["AnyPeftConfig"]


AnyPeftConfig = Annotated[
    LoRAConfig | FullTuneConfig | PeftStackConfig,
    Field(discriminator="kind"),
]
"""Any valid PEFT configuration, discriminated by the ``kind`` field."""
