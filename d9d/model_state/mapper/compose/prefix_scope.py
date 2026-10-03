import torch

from d9d.model_state.mapper import ModelStateMapper, StateGroup


def _build_groups(mapper: ModelStateMapper, source_prefix: str, target_prefix: str) -> frozenset[StateGroup]:
    groups = set()
    for group in mapper.state_dependency_groups():
        scoped_inputs = frozenset(f"{source_prefix}{k}" for k in group.inputs)
        scoped_outputs = frozenset(f"{target_prefix}{k}" for k in group.outputs)
        groups.add(StateGroup(inputs=scoped_inputs, outputs=scoped_outputs))

    return frozenset(groups)


class ModelStateMapperPrefixScope(ModelStateMapper):
    """Mapper that runs a child mapper under key prefixes.

    Use it to apply a mapper written for a submodule (e.g. one operating on ``"in_proj"``) to the state dict of
    a parent module. Input (source) and output (target) prefixes are independent.
    """

    def __init__(self, mapper: ModelStateMapper, source_prefix: str = "", target_prefix: str = "") -> None:
        """Constructs the ``ModelStateMapperPrefixScope`` object.

        Args:
            mapper: The child mapper to run within the scope.
            source_prefix: The prefix added to the input keys of ``mapper``.
            target_prefix: The prefix added to the output keys of ``mapper``.
        """
        self._mapper = mapper
        self._source_prefix = source_prefix
        self._target_prefix = target_prefix
        self._groups = _build_groups(mapper, source_prefix, target_prefix)

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return self._groups

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scoped_group = {k.removeprefix(self._source_prefix): v for k, v in group.items()}

        scoped_result = self._mapper.apply(scoped_group)

        return {f"{self._target_prefix}{k}": v for k, v in scoped_result.items()}
