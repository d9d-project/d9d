import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperStackTensors(ModelStateMapper):
    """
    Stacks multiple input tensors into a single output tensor producing
    a new stacking dimension.
    """

    def __init__(
        self,
        source_names: list[str],
        target_name: str,
        dim: int,
    ) -> None:
        """
        Constructs ModelStateMapperStackTensors object.

        Args:
            source_names: Names of the input tensors to read.
            target_name: Name of the resulting stacked output tensor.
            dim: Dimension along which to stack the tensors.
        """
        self._source_names = source_names
        self._target_name = target_name
        self._stack_dim = dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset(self._source_names), outputs=frozenset([self._target_name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source_tensors = [group[name] for name in self._source_names]
        out_tensor = torch.stack(source_tensors, dim=self._stack_dim)

        return {self._target_name: out_tensor.contiguous()}


class ModelStateMapperUnstackTensors(ModelStateMapper):
    """
    Unstacks a single input tensor into multiple output tensors along a specified
    dimension.
    """

    def __init__(
        self,
        source_name: str,
        target_names: list[str],
        dim: int,
    ) -> None:
        """
        Constructs ModelStateMapperUnstackTensors object.

        Args:
            source_name: Name of the input tensor to read.
            target_names: Names of the resulting unstacked output tensors.
            dim: Dimension along which to unstack the tensor.
        """
        self._source_name = source_name
        self._target_names = target_names
        self._unstack_dim = dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._source_name]), outputs=frozenset(self._target_names))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tensor = group[self._source_name]
        unstacked_tensors = torch.unbind(tensor, dim=self._unstack_dim)

        return {
            name: unstacked_tensor.contiguous()
            for name, unstacked_tensor in zip(self._target_names, unstacked_tensors, strict=True)
        }


class ModelStateMapperChunkTensors(ModelStateMapper):
    """
    Chunks a single input tensor into multiple output tensors along a specified dimension.
    """

    def __init__(
        self,
        source_name: str,
        target_names: list[str],
        dim: int,
    ) -> None:
        """
        Constructs ModelStateMapperChunkTensors object.

        Args:
            source_name: Name of the input tensor to read.
            target_names: Names of the resulting chunked output tensors.
            dim: Dimension along which to chunk the tensor.
        """
        self._source_name = source_name
        self._target_names = target_names
        self._chunk_dim = dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset([self._source_name]), outputs=frozenset(self._target_names))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tensor = group[self._source_name]
        chunked_tensors = torch.chunk(tensor, chunks=len(self._target_names), dim=self._chunk_dim)

        return {
            name: chunked_tensor.contiguous()
            for name, chunked_tensor in zip(self._target_names, chunked_tensors, strict=True)
        }


class ModelStateMapperConcatenateTensors(ModelStateMapper):
    """
    Concatenates ('unchunks') multiple input tensors into a single output tensor
    along a specified dimension.
    """

    def __init__(
        self,
        source_names: list[str],
        target_name: str,
        dim: int,
    ) -> None:
        """
        Constructs ModelStateMapperUnchunkTensors object.

        Args:
            source_names: Names of the input tensors to read.
            target_name: Name of the resulting concatenated (unchunked) output tensor.
            dim: Dimension along which to concatenate the tensors.
        """
        self._source_names = source_names
        self._target_name = target_name
        self._chunk_dim = dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset(self._source_names), outputs=frozenset([self._target_name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source_tensors = [group[name] for name in self._source_names]
        out_tensor = torch.cat(source_tensors, dim=self._chunk_dim)

        return {self._target_name: out_tensor.contiguous()}
