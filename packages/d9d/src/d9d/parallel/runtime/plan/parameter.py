"""The parameters of each parameter group of a stage."""

import dataclasses
from collections.abc import Sequence

from torch import Tensor, nn
from torch.distributed.tensor import DTensor


@dataclasses.dataclass(slots=True, eq=False)
class ParameterPlan:
    """The plan of one parameter: the parameter and the modules that hold it.

    Attributes:
        name: The name of the parameter in the stage root, e.g. ``layers.3.attn.q_proj.weight``. A tied parameter
            has its first name, as in the state dict.
        owners: The modules that hold the parameter and its name in each. A tied parameter has several.
        sharded: The parameter that the optimizer and checkpoints see: a ``DTensor``, or a plain tensor outside a
            device mesh.
        trainable: Whether the job computes the gradient of the parameter.
    """

    name: str
    owners: list[tuple[nn.Module, str]]
    sharded: nn.Parameter
    trainable: bool


def local_tensor(data: Tensor) -> Tensor:
    """Returns the local part of a ``DTensor``, or the tensor itself.

    Args:
        data: A ``DTensor`` or a plain tensor.

    Returns:
        The tensor that this rank holds.
    """
    return data.to_local() if isinstance(data, DTensor) else data


def _group_modules(root: nn.Module, roots: set[nn.Module]) -> list[nn.Module]:
    result: dict[nn.Module, None] = {}
    # A loop instead of a nested recursive function, which would hold the modules in a reference cycle.
    pending = [root]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result[current] = None
        pending.extend(child for child in reversed(list(current.children())) if child not in roots)
    return list(result)


def plan_parameters(groups: Sequence[nn.Module], training: bool) -> list[list[ParameterPlan]]:
    """Plans the parameters that each group holds.

    A group holds the parameters of its root and of the root's descendants, except those of other roots. A tied
    parameter has one plan, listed in every group that holds it.

    Args:
        groups: The roots of the groups of a stage. The first one is the stage root.
        training: Whether the job computes gradients.

    Returns:
        The parameters of each group, by group index.

    Raises:
        ValueError: If a trainable parameter has no ``grad_dtype``.
    """
    names: dict[int, str] = {}
    for name, param in groups[0].named_parameters(remove_duplicate=False):
        names.setdefault(id(param), name)

    known: dict[int, ParameterPlan] = {}
    # A module can belong to several groups, but each of its slots holds the parameter once.
    slots: set[tuple[nn.Module, str]] = set()
    plans: list[list[ParameterPlan]] = []

    for group in groups:
        group_plans: dict[int, ParameterPlan] = {}
        for submodule in _group_modules(group, set(groups)):
            for name, param in submodule.named_parameters(recurse=False, remove_duplicate=False):
                if id(param) not in known:
                    known[id(param)] = ParameterPlan(
                        name=names[id(param)], owners=[], sharded=param, trainable=training and param.requires_grad
                    )
                plan = known[id(param)]

                if (submodule, name) not in slots:
                    slots.add((submodule, name))
                    plan.owners.append((submodule, name))
                group_plans[id(param)] = plan
        plans.append(list(group_plans.values()))

    for plan in known.values():
        # PyTorch accepts a gradient of any dtype with grad_dtype None, but a gradient buffer needs one dtype.
        if plan.trainable and plan.sharded.grad_dtype is None:
            raise ValueError(
                f"The trainable parameter ({plan.name}) has grad_dtype None, so the runtime cannot allocate its "
                "gradient buffer. Set grad_dtype to a dtype, e.g. param.grad_dtype = param.dtype."
            )

    return plans
