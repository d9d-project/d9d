import abc
import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class StateGroup:
    """Represents an atomic unit of dependency in the model state transformation graph.

    A ``StateGroup`` binds a set of input keys (source) to a set of output keys (destination).

    Attributes:
        inputs: The complete set of keys required from the source state dictionary to satisfy this dependency.
        outputs: The complete set of keys that will be produced as a result of this transformation.
    """

    inputs: frozenset[str]
    outputs: frozenset[str]


class ModelStateMapper(abc.ABC):
    """Base class for all model state transformations.

    A mapper separates the declaration of a transformation from its execution:

    1.  Declarative (topology): ``state_dependency_groups()`` announces *what* the mapper does without touching
        data. This lets the system build execution graphs, validate chains, detect collisions and shard work
        *before* it allocates memory.
    2.  Imperative (execution): ``apply()`` runs the PyTorch operations on model states.
    """

    @abc.abstractmethod
    def state_dependency_groups(self) -> frozenset[StateGroup]:
        """Returns the set of independent dependency groups this mapper handles.

        Returns:
            A frozenset of ``StateGroup`` objects. Each group is a disjoint operation. For example, a mapper that
            renames ten independent tensors returns ten distinct groups, so they can be sharded or processed
            individually.
        """
        ...

    @abc.abstractmethod
    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Executes the transformation on the tensors of one dependency group.

        The caller guarantees that ``group`` contains all keys listed in ``inputs`` of the active ``StateGroup``.
        Implementations must return all keys listed in its ``outputs``.

        Args:
            group: The source tensors. Keys match ``StateGroup.inputs``.

        Returns:
            The transformed tensors. Keys must exactly match ``StateGroup.outputs``.
        """
        ...
