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


class ModelStateMapperSqueeze(ModelStateMapper):
    """
    Squeezes an input tensor along a specified dimension or all dimensions of size 1.
    """

    def __init__(
        self,
        name: str,
        dim: int | None = None,
    ) -> None:
        """
        Constructs ModelStateMapperSqueeze object.

        Args:
            name: Name of the tensor to operate on.
            dim: Dimension to squeeze. If not provided, squeezes all dimensions of size 1.
        """
        self._name = name
        self._dim = dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._name]), outputs=frozenset([self._name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tensor = group[self._name]

        if self._dim is not None:
            squeezed_tensor = tensor.squeeze(self._dim)
        else:
            squeezed_tensor = tensor.squeeze()

        return {self._name: squeezed_tensor}


class ModelStateMapperUnsqueeze(ModelStateMapper):
    """
    Unsqueezes an input tensor along a specified dimension.
    """

    def __init__(
        self,
        name: str,
        dim: int,
    ) -> None:
        """
        Constructs ModelStateMapperUnsqueeze object.

        Args:
            name: Name of the tensor to operate on.
            dim: Dimension to unsqueeze.
        """
        self._name = name
        self._dim = dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._name]), outputs=frozenset([self._name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {self._name: group[self._name].unsqueeze(self._dim)}
