import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperSelectChildModules(ModelStateMapper):
    """
    Selects a set of keys belonging to a specific parent module (prefix) and
    renames them by removing that prefix.

    This is effectively a batch rename operation that "hoists" parameters
    from a submodule scope to the current scope.
    """

    def __init__(self, base_names: list[str], parent_name: str):
        self._base_names = base_names
        self._parent_prefix = f"{parent_name}."

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([
            StateGroup(
                inputs=frozenset([self._parent_prefix + name]),
                outputs=frozenset([name])
            )
            for name in self._base_names
        ])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        name, value = next(iter(group.items()))
        if name.startswith(self._parent_prefix):
            return {
                name[len(self._parent_prefix):]: value
            }
        else:
            return {

            }
