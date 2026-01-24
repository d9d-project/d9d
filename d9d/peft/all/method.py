from typing import Self, cast

from pydantic import BaseModel
from torch import nn

from ..all.config import PeftStackConfig
from ..base import PeftInjectionResult, PeftMethod, TConfig
from ..full_tune.config import FullTuneConfig
from ..full_tune.method import FullTune
from ..lora.config import LoRAConfig
from ..lora.method import LoRA


class PeftStack(PeftMethod[PeftStackConfig]):
    """
    A composite PEFT method that applies a list of methods sequentially.
    """

    def __init__(self, methods: list[PeftMethod]):
        """
        Constructs a PeftStack object.

        Args:
            methods: A list of instantiated PEFT methods to apply in order.
        """

        self._methods = methods

    def inject(self, module: nn.Module) -> PeftInjectionResult:
        params_to_train = []
        state_mappers = []

        for method in self._methods:
            result = method.inject(module)
            params_to_train.extend(result.parameters_to_train)
            state_mappers.extend(result.load_state_mappers)

        return PeftInjectionResult(
            parameters_to_train=params_to_train,
            load_state_mappers=state_mappers
        )

    def merge(self, module: nn.Module):
        for method in self._methods[::-1]:
            method.merge(module)

    @classmethod
    def from_config(cls, config: PeftStackConfig) -> Self:
        methods = []

        for method in config.methods:
            methods.append(peft_method_from_config(method))

        return cls(methods)


_PEFT_CONFIG_MAP: dict[type[BaseModel], type[PeftMethod]] = {
    LoRAConfig: LoRA,
    FullTuneConfig: FullTune,
    PeftStackConfig: PeftStack
}


def peft_method_from_config(config: TConfig) -> PeftMethod[TConfig]:
    """
    Factory function to instantiate the correct PeftMethod based on the configuration type.

    Args:
        config: A specific PEFT configuration object (e.g., LoRAConfig).

    Returns:
        The corresponding method instance.
    """

    method_cls = cast(type[PeftMethod[TConfig]], _PEFT_CONFIG_MAP[type(config)])
    return method_cls.from_config(config)
