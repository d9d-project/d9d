import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperSelectChildModules(ModelStateMapper):
    """Mapper that selects the keys of a child module and strips the module prefix.

    It is a batch rename that moves parameters from a submodule scope to the current scope.
    """

    def __init__(self, base_names: list[str], parent_name: str):
        """Constructs the ``ModelStateMapperSelectChildModules`` object.

        Args:
            base_names: The keys relative to the child module, e.g. ``"weight"``.
            parent_name: The name of the child module, e.g. ``"in_proj"``. Input keys are
                ``"{parent_name}.{base_name}"``.
        """
        self._base_names = base_names
        self._parent_prefix = f"{parent_name}."

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset(
            [
                StateGroup(inputs=frozenset([self._parent_prefix + name]), outputs=frozenset([name]))
                for name in self._base_names
            ]
        )

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        name, value = next(iter(group.items()))
        if name.startswith(self._parent_prefix):
            return {name[len(self._parent_prefix) :]: value}
        else:
            return {}
