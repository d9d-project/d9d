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
    """Encapsulates a child mapper and isolates its execution by applying string prefixes.

    This mapper allows a child mapper designed for a specific submodule
    (e.g., operating on "in_proj") to be seamlessly integrated into a larger
    parent module's state dictionary by virtually scoping its operations
    using completely independent input (source) and output (target) prefixes.
    """

    def __init__(self, mapper: ModelStateMapper, source_prefix: str = "", target_prefix: str = "") -> None:
        """Constructs a ModelStateMapperPrefixScope object.

        Args:
            mapper: The encapsulated child mapper to execute within the scope.
            source_prefix: The string prefix defining the scope boundary for incoming state keys.
            target_prefix: The string prefix defining the scope boundary for outgoing state keys.
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
