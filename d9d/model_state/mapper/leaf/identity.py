import torch

from d9d.model_state.mapper.abc import StateGroup, ModelStateMapper


class ModelStateMapperIdentity(ModelStateMapper):
    """
    Passes a single state tensor through unchanged.
    """

    def __init__(self, name: str):
        self._name = name

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([
            StateGroup(
                inputs=frozenset([self._name]),
                outputs=frozenset([self._name])
            )
        ])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return group
