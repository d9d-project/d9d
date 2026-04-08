import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperIdentity(ModelStateMapper):
    """
    Passes a single state tensor through unchanged.
    """

    def __init__(self, name: str):
        self._name = name

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._name]), outputs=frozenset([self._name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return group


class ModelStateMapperTranspose(ModelStateMapper):
    """
    Transposes an input tensor along two specified dimensions.
    """

    def __init__(
        self,
        name: str,
        dims: tuple[int, int],
    ) -> None:
        """
        Constructs ModelStateMapperTranspose object.

        Args:
            name: Name of the tensor to operate on.
            dims: Dimensions to transpose.
        """
        self._name = name
        self._dims = dims

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._name]), outputs=frozenset([self._name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {self._name: group[self._name].transpose(*self._dims).contiguous()}
