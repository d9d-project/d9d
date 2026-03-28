import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperStackTensors(ModelStateMapper):
    """
    Stacks multiple input tensors into a single output tensor producing
    a new stacking dimension. Optionally transposes the resulting tensor.
    """

    def __init__(
        self,
        source_names: list[str],
        target_name: str,
        stack_dim: int,
        transpose_dims: tuple[int, int] | None = None,
    ) -> None:
        """
        Constructs ModelStateMapperStackTensors object.

        Args:
            source_names: Names of the input tensors to read.
            target_name: Name of the resulting stacked output tensor.
            stack_dim: Dimension along which to stack the tensors.
            transpose_dims: Optional tuple of two dimensions to transpose after stacking.
        """
        self._source_names = source_names
        self._target_name = target_name
        self._stack_dim = stack_dim
        self._transpose_dims = transpose_dims

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset(self._source_names), outputs=frozenset([self._target_name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source_tensors = [group[name] for name in self._source_names]
        out_tensor = torch.stack(source_tensors, dim=self._stack_dim)

        if self._transpose_dims is not None:
            out_tensor = out_tensor.transpose(*self._transpose_dims)

        return {self._target_name: out_tensor.contiguous()}


class ModelStateMapperUnstackTensors(ModelStateMapper):
    """
    Unstacks a single input tensor into multiple output tensors along a specified
    dimension. Optionally transposes the input tensor before unstacking (acting as the
    exact inverse of stacking with a post-transpose).
    """

    def __init__(
        self,
        source_name: str,
        target_names: list[str],
        unstack_dim: int,
        transpose_dims: tuple[int, int] | None = None,
    ) -> None:
        """
        Constructs ModelStateMapperUnstackTensors object.

        Args:
            source_name: Name of the input tensor to read.
            target_names: Names of the resulting unstacked output tensors.
            unstack_dim: Dimension along which to unstack the tensor.
            transpose_dims: Optional tuple of two dimensions to transpose before unstacking.
        """
        self._source_name = source_name
        self._target_names = target_names
        self._unstack_dim = unstack_dim
        self._transpose_dims = transpose_dims

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._source_name]), outputs=frozenset(self._target_names))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tensor = group[self._source_name]

        if self._transpose_dims is not None:
            tensor = tensor.transpose(*self._transpose_dims)

        unstacked_tensors = torch.unbind(tensor, dim=self._unstack_dim)

        return {
            name: unstacked_tensor.contiguous()
            for name, unstacked_tensor in zip(self._target_names, unstacked_tensors, strict=True)
        }
