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
    """The buckets that one backward pass reduces after each of its segments.

    Attributes:
        final: Whether this is the last backward pass of the stage in the step. The final pass also sums the
            gradients across replicas.
        after_segment: The buckets to reduce after each segment, by segment index, in reduction order.
    """

    final: bool
    after_segment: Mapping[int, tuple[BucketId, ...]]
