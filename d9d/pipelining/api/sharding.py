import dataclasses

from d9d.core.sharding import ShardingSpec


@dataclasses.dataclass
class PipelineShardingSpec:
    input_data: ShardingSpec | None = None
    input_kwargs: ShardingSpec | None = None
