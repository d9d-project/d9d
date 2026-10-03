from typing import Self

import torch
from torch import nn

from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.leaf import ModelStateMapperRename
from d9d.module.block.moe import GroupedLinear

from ..base import PeftInjectionResult, PeftMethod
from .config import LoRAConfig
from .layer import LoRAGroupedLinear, LoRALinear

_CAN_APPLY_MODULES = (nn.Linear, GroupedLinear)
_LORA_MODULES = (LoRALinear, LoRAGroupedLinear)


def named_modules_without_lora(
    module: nn.Module, memo: set[nn.Module] | None = None, prefix: str = "", remove_duplicate: bool = True
):
    """Yields named modules, skipping submodules that are already LoRA layers.

    This prevents injecting LoRA into an already wrapped layer again. Mirrors ``nn.Module.named_modules``.

    Args:
        module: The root module to traverse.
        memo: Set of visited modules.
        prefix: Current name prefix.
        remove_duplicate: Whether to skip modules that are already in ``memo``.

    Yields:
        Pairs of ``(name, module)``.
    """
    if isinstance(module, _LORA_MODULES):
        return

    if memo is None:
        memo = set()
    if module in memo:
        return

    if remove_duplicate:
        memo.add(module)

    yield prefix, module

    for name, submodule in module.named_children():
        if submodule is None:
            continue

        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from named_modules_without_lora(submodule, memo, submodule_prefix, remove_duplicate)


class LoRA(PeftMethod[LoRAConfig]):
    """Implements Low-Rank Adaptation (LoRA).

    ``inject`` replaces every ``nn.Linear`` or ``GroupedLinear`` layer whose name matches the configured pattern
    with a LoRA wrapper. The original ``layer.weight`` moves to ``layer.base.weight``. For each wrapped layer,
    ``inject`` returns a ``ModelStateMapperRename``, so standard checkpoints still load.
    """

    def __init__(self, config: LoRAConfig):
        """Constructs the ``LoRA`` object.

        Args:
            config: LoRA configuration with the module name pattern and hyperparameters.
        """
        self._config = config

    def inject(self, module: nn.Module) -> PeftInjectionResult:
        params_to_train: list[nn.Parameter] = []
        state_mappers: list[ModelStateMapper] = []

        for mod_name, mod in named_modules_without_lora(module):
            if not isinstance(mod, _CAN_APPLY_MODULES):
                continue

            if not self._config.module_name_pattern.fullmatch(mod_name):
                continue

            lora_mod: LoRALinear | LoRAGroupedLinear
            if isinstance(mod, nn.Linear):
                lora_mod = LoRALinear(mod, self._config.params)
            elif isinstance(mod, GroupedLinear):
                lora_mod = LoRAGroupedLinear(mod, self._config.params)
            else:
                raise ValueError(f"LoRA does not support layer type ({type(mod).__name__}).")

            params_to_train.extend(lora_mod.lora_A.parameters())
            params_to_train.extend(lora_mod.lora_B.parameters())

            state_mappers.append(
                ModelStateMapperRename(name_from=f"{mod_name}.weight", name_to=f"{mod_name}.base.weight")
            )

            module.set_submodule(mod_name, lora_mod)

        return PeftInjectionResult(parameters_to_train=params_to_train, load_state_mappers=state_mappers)

    def merge(self, module: nn.Module):
        for mod_name, mod in module.named_modules():
            if not isinstance(mod, _LORA_MODULES):
                continue

            if not self._config.module_name_pattern.fullmatch(mod_name):
                continue

            with torch.no_grad():
                orig_mod = mod.merge_with_base_()

            module.set_submodule(mod_name, orig_mod)

    @classmethod
    def from_config(cls, config: LoRAConfig) -> Self:
        return cls(config)
