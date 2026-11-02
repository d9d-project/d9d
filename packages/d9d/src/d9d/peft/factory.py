from .base import PeftMethod
from .full_tune import FullTune, FullTuneConfig
from .lora import LoRA, LoRAConfig
from .stack import PeftStack, PeftStackConfig
from .stack.config import AnyPeftConfig


def peft_method_from_config(config: AnyPeftConfig) -> PeftMethod:
    """Creates the PEFT method that the ``kind`` field of the configuration selects.

    Args:
        config: The PEFT configuration.

    Returns:
        The PEFT method.

    Raises:
        ValueError: If the configuration type is unknown.
    """
    match config:
        case LoRAConfig():
            return LoRA.from_config(config)
        case FullTuneConfig():
            return FullTune.from_config(config)
        case PeftStackConfig():
            return PeftStack.from_config(config)
        case _:
            raise ValueError(f"PEFT configuration type ({type(config).__name__}) is not supported.")
