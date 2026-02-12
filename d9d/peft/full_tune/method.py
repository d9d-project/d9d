from typing import Self

from torch import nn

from ..base import PeftInjectionResult, PeftMethod
from .config import FullTuneConfig


class FullTune(PeftMethod[FullTuneConfig]):
    """
    Implements Full Fine-Tuning as a 'PEFT' method.

    Instead of injecting adapters, this method simply identifies existing parameters
    that match the configuration pattern and marks them for training.
    """

    def __init__(self, config: FullTuneConfig):
        """
        Constructs a FullTune object.

        Args:
            config: Configuration defining the module name patterns to fine-tune.
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
        pass  # do nothing here

    @classmethod
    def from_config(cls, config: FullTuneConfig) -> Self:
        return cls(config)
