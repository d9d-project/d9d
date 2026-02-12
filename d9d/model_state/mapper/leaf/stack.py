import torch

from d9d.model_state.mapper.abc import ModelStateMapper, StateGroup


class ModelStateMapperStackTensors(ModelStateMapper):
    """
    Stacks multiple input tensors with names `source_names` into a single output tensor with name `target_name`
    producing new `stack_dim` dimension.
    """

    def __init__(self, source_names: list[str], target_name: str, stack_dim: int):
        self._source_names = source_names
        self._target_name = target_name
        self._stack_dim = stack_dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=frozenset(self._source_names), outputs=frozenset([self._target_name]))])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        source_tensors = [group[name] for name in self._source_names]
        return {self._target_name: torch.stack(source_tensors, dim=self._stack_dim)}
