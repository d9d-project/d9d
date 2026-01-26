import dataclasses

from d9d.core.types import PyTree


@dataclasses.dataclass(slots=True, frozen=True)
class SpecReplicate:
    """
    Specifies that a leaf node should be replicated across all shards.
    """


@dataclasses.dataclass(slots=True, frozen=True)
class SpecShard:
    """
    Specifies that a leaf node should be split along a specific dimension.

    Attributes:
        dim: The dimension to split.
        do_stack: If True, sharding will squeeze the sharded dimension (it should be exactly the num_shards length)
    """

    dim: int
    do_stack: bool = False


ShardingSpecLeaf = SpecReplicate | SpecShard
ShardingSpec = PyTree[ShardingSpecLeaf]
