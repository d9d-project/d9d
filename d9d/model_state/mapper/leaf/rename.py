import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperRename(ModelStateMapper):
    """Renames a single state tensor."""

    def __init__(self, name_from: str, name_to: str):
        """Constructs the ``ModelStateMapperRename`` object.

        Args:
            name_from: The source key.
            name_to: The target key.
        """
        self._name_from = name_from
        self._name_to = name_to

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._name_from]), outputs=frozenset([self._name_to]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {self._name_to: group[self._name_from]}
