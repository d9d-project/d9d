"""The plans of the stages: their groups, parameters and buckets."""

import dataclasses
from collections.abc import Mapping, Sequence

from torch import nn

from .bucket import BucketPlan, plan_buckets
from .parameter import ParameterPlan, plan_parameters


@dataclasses.dataclass(slots=True)
class StagePlan:
    """The plan of one stage: its groups, parameters and buckets.

    Attributes:
        groups: The roots of the groups of the stage. The first one is the stage root.
        parameters: The parameters of the stage, each once.
        parameters_by_group: The parameters that each group holds, by group index.
        buckets: The buckets in reduction order.
    """

    groups: Sequence[nn.Module]
    parameters: list[ParameterPlan]
    parameters_by_group: list[list[ParameterPlan]]
    buckets: list[BucketPlan]


def plan_stages(
    stages: Mapping[int, Sequence[nn.Module]],
    training: bool,
    bucket_size_mib: int,
    gather_dims: frozenset[str],
) -> dict[int, StagePlan]:
    """Plans the parameters and buckets of every stage.

    Args:
        stages: The roots of the groups of each stage, by stage index. The first root is the stage root.
        training: Whether the job computes gradients.
        bucket_size_mib: The size limit of buckets without a gather, in MiB.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The plan of each stage.

    Raises:
        ValueError: If a parameter is shared by several stages, if a trainable parameter has no ``grad_dtype``, if
            the placements of a parameter are not supported, or if a shard does not match ``torch.chunk``.
    """
    plans: dict[int, StagePlan] = {}
    parameter_stages: dict[int, int] = {}

    for stage_index, groups in stages.items():
        parameters_by_group = plan_parameters(groups, training)
        parameters = list(dict.fromkeys(x for group in parameters_by_group for x in group))
        for parameter in parameters:
            owner = parameter_stages.setdefault(id(parameter.sharded), stage_index)
            if owner != stage_index:
                raise ValueError(
                    f"The parameter ({parameter.name}) is shared by stages ({owner}) and ({stage_index}). Place tied "
                    "parameters on one stage."
                )

        plans[stage_index] = StagePlan(
            groups=groups,
            parameters=parameters,
            parameters_by_group=parameters_by_group,
            buckets=plan_buckets(parameters_by_group, bucket_size_mib, gather_dims),
        )

    return plans
