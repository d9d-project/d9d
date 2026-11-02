from typing import Self

from torch import nn

from ..base import PeftInjectionResult, PeftMethod
from .config import PeftStackConfig


class PeftStack(PeftMethod[PeftStackConfig]):
    """PEFT method that applies a list of PEFT methods in order."""

    def __init__(self, methods: list[PeftMethod]):
        """Constructs the ``PeftStack`` object.

        Args:
            methods: The PEFT methods to apply in order. ``merge`` runs them in reverse order.
        """
        self._methods = methods

    def inject(self, module: nn.Module) -> PeftInjectionResult:
        params_to_train = []
        state_mappers = []

        for method in self._methods:
            result = method.inject(module)
            params_to_train.extend(result.parameters_to_train)
            state_mappers.extend(result.load_state_mappers)

        return PeftInjectionResult(parameters_to_train=params_to_train, load_state_mappers=state_mappers)

    def merge(self, module: nn.Module):
        for method in self._methods[::-1]:
            method.merge(module)

    @classmethod
    def from_config(cls, config: PeftStackConfig) -> Self:
        # The factory imports this module to build a stack.
        from ..factory import peft_method_from_config  # noqa: PLC0415 - circular import

        return cls([peft_method_from_config(method) for method in config.methods])
