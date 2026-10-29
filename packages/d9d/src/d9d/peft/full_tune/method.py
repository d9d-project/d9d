from typing import Self

from torch import nn

from ..base import PeftInjectionResult, PeftMethod
from .config import FullTuneConfig


class FullTune(PeftMethod[FullTuneConfig]):
    """PEFT method for full fine-tuning.

    It injects no adapters. It marks all parameters of the matching modules as trainable.
    """

    def __init__(self, config: FullTuneConfig):
        """Constructs the ``FullTune`` object.

        Args:
            config: Configuration with the module name pattern to fine-tune.
        """
        self._config = config

    def inject(self, module: nn.Module) -> PeftInjectionResult:
        params_to_train = []

        for mod_name, mod in module.named_modules():
            is_applicable = self._config.module_name_pattern.fullmatch(mod_name)

            if is_applicable:
                params_to_train.extend(mod.parameters())

        return PeftInjectionResult(parameters_to_train=params_to_train, load_state_mappers=[])

    def merge(self, module: nn.Module):
        # Full fine-tuning updates the original parameters, so there is nothing to merge.
        pass

    @classmethod
    def from_config(cls, config: FullTuneConfig) -> Self:
        return cls(config)
