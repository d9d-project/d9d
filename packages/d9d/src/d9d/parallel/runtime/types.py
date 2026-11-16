"""Identifiers of gradient reductions, shared by the runtime and the pipeline program."""

import dataclasses
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True, slots=True)
class BucketId:
    """A bucket of one pipeline stage.

    Attributes:
        stage: The index of the stage.
        index: The index of the bucket in the stage, in reduction order.
    """

    stage: int
    index: int


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class ReductionPlan:
    """The buckets that one backward pass reduces after each of its regions.

    Attributes:
        final: Whether this is the last backward pass of the stage in the step. The final pass also sums the
            gradients across replicas.
        after_region: The buckets to reduce after each region occurrence, in reduction order.
        before_region: The gathered buckets whose gradient buffers must be prepared before each region occurrence.
    """

    final: bool
    after_region: Mapping[int, tuple[BucketId, ...]]
    before_region: Mapping[int, tuple[BucketId, ...]] = dataclasses.field(default_factory=dict)
