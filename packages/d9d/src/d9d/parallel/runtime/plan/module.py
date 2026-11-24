import dataclasses
from collections.abc import Mapping, Sequence

from torch import nn

from .bucket import BucketPlan, plan_buckets
from .parameter import GroupPlan, ParameterPlan, plan_groups


@dataclasses.dataclass(slots=True)
class ModulePlan:
    """The plan of one registered module: its groups, parameters and buckets.

    Attributes:
        groups: The parameter groups, by group index.
        parameters: The parameters of the module, each once.
        buckets: The buckets in reduction order.
    """

    groups: list[GroupPlan]
    parameters: list[ParameterPlan]
    buckets: list[BucketPlan]


def plan_modules(
    modules: Mapping[nn.Module, Sequence[nn.Module]],
    training: bool,
    bucket_size_mib: int,
    gather_dims: frozenset[str],
) -> dict[nn.Module, ModulePlan]:
    """Plans the parameters and buckets of every registered module.

    Args:
        modules: The roots of the other groups of each registered module, by module.
        training: Whether the job computes gradients.
        bucket_size_mib: The size limit of buckets without a gather, in MiB.
        gather_dims: The mesh dimensions along which data parallelism shards parameters.

    Returns:
        The plan of each module.

    Raises:
        ValueError: If a parameter is shared by several modules, if a trainable parameter has no ``grad_dtype``, if
            the placements of a parameter are not supported, or if a shard does not match ``torch.chunk``.
    """
    plans: dict[nn.Module, ModulePlan] = {}
    # The first registered module of each parameter and the path of the parameter in it.
    owners: dict[int, tuple[nn.Module, str]] = {}

    for module, group_roots in modules.items():
        groups = plan_groups(module, group_roots, training)
        parameters = list(dict.fromkeys(x for group in groups for x in group.parameters))
        for parameter in parameters:
            owner, path = owners.setdefault(id(parameter.sharded), (module, parameter.name))
            if owner is not module:
                raise ValueError(
                    f"The parameter ({parameter.name}) of the module ({type(module).__name__}) is the parameter "
                    f"({path}) of the module ({type(owner).__name__}) too. Place tied parameters in one registered "
                    "module."
                )

        plans[module] = ModulePlan(
            groups=groups,
            parameters=parameters,
            buckets=plan_buckets(groups, bucket_size_mib, gather_dims),
        )

    return plans
