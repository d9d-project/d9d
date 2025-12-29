import abc
import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class StateGroup:
    """
    Represents an atomic unit of dependency in the model state transformation graph.

    A `StateGroup` defines a strict contract between a set of input keys (source)
    and a set of output keys (destination).

    Attributes:
        inputs: The complete set of keys required from the source state dictionary to satisfy this dependency.
        outputs: The complete set of keys that will be produced as a result of this transformation.
    """

    inputs: frozenset[str]
    outputs: frozenset[str]


class ModelStateMapper(abc.ABC):
    """
   The abstract base class for all model state transformation operations.

   This class serves as the interface between the definition of a transformation
   topology and the actual execution of tensor operations.

   It enforces a Declarative vs. Imperative separation of concerns:

   1.  Declarative (Topology): Through `state_dependency_groups()`, the mapper
       announces *what* it intends to do without handling any data. This allows the system to build execution graphs,
       validate chains, detect collisions, and shard tasks *before* allocating memory.
   2.  Imperative (Execution): Through `apply()`, the mapper performs the
       actual logic (PyTorch operations) on model states.
   """

    @abc.abstractmethod
    def state_dependency_groups(self) -> frozenset[StateGroup]:
        """
        Calculates and returns the set of independent dependency groups this mapper handles.

        Returns:
            A frozenset of `StateGroup` objects. Each group
            represents a disjoint operation. For example, a mapper that renames ten
            independent tensors would return ten distinct `StateGroup` objects,
            allowing them to be sharded or processed individually.
        """
        ...

    @abc.abstractmethod
    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Executes the transformation logic on a specific dictionary of tensors.

        The orchestration system guarantees that the `group` dictionary passed here contains
        all keys listed in the `inputs` of the active `StateGroup`.

        Implementation of this method should guarantee that the result will contain all keys listed in the `outputs`.

        Args:
           group: A dictionary containing the source data.
               Keys match `StateGroup.inputs`.

        Returns:
           A dictionary containing the transformed data. Keys must strictly match `StateGroup.outputs`.
        """
        ...
