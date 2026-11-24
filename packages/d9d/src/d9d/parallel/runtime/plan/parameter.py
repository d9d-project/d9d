import dataclasses
from collections.abc import Sequence

from torch import Tensor, nn
from torch.distributed.tensor import DTensor


@dataclasses.dataclass(slots=True, eq=False)
class ParameterPlan:
    """The plan of one parameter: the parameter, its name and the modules that hold it.

    Attributes:
        name: The path of the parameter in the registered module, e.g. ``layers.3.attn.q_proj.weight``. A tied
            parameter has its first path, as in the state dict.
        owners: Each module that holds the parameter, with its name there. A tied parameter has several.
        sharded: The parameter that the optimizer and checkpoints see: a ``DTensor``, or a plain tensor in a job
            without parallelism.
        trainable: Whether the job computes the gradient of the parameter.
    """

    name: str
    owners: list[tuple[nn.Module, str]]
    sharded: nn.Parameter
    trainable: bool


@dataclasses.dataclass(slots=True)
class GroupPlan:
    """The plan of one parameter group: its root and the parameters that it holds.

    Attributes:
        root: The root module of the group.
        name: The path of the root in the registered module, e.g. ``layers.3``, or the class name of the
            registered module for its own group.
        parameters: The parameters of the group.
    """

    root: nn.Module
    name: str
    parameters: list[ParameterPlan]


def module_path(root: nn.Module, module: nn.Module) -> str:
    """Returns the name of a module in errors: its path in a root module, or its class name outside of it.

    Args:
        root: The root module, e.g. a registered module.
        module: The module to name.

    Returns:
        The path of ``module`` in ``root``, e.g. ``layers.3``, or the class name for ``root`` itself and for a module
        outside it.
    """
    for path, submodule in root.named_modules(remove_duplicate=False):
        if submodule is module and path:
            return path
    return type(module).__name__


def local_tensor(data: Tensor) -> Tensor:
    """Returns the local tensor of a ``DTensor``, or the tensor itself.

    Args:
        data: A ``DTensor`` or a plain tensor.

    Returns:
        The tensor on this rank.
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


def plan_groups(module: nn.Module, group_roots: Sequence[nn.Module], training: bool) -> list[GroupPlan]:
    """Plans the parameter groups of a registered module.

    A group holds the parameters of its root and of the root's descendants, except those of other roots. A tied
    parameter has one plan in every group that holds it.

    Args:
        module: The registered module, the root of its own group.
        group_roots: The roots of the other groups.
        training: Whether the job computes gradients.

    Returns:
        The group of ``module``, then the groups of ``group_roots`` in their order.

    Raises:
        ValueError: If a trainable parameter has no ``grad_dtype``.
    """
    names: dict[int, str] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        names.setdefault(id(param), name)

    known: dict[int, ParameterPlan] = {}
    # A module can belong to several groups, but each of its slots holds the parameter once.
    slots: set[tuple[nn.Module, str]] = set()
    groups: list[GroupPlan] = []
    roots = [module, *group_roots]

    for root in roots:
        group_plans: dict[int, ParameterPlan] = {}
        for submodule in _group_modules(root, set(roots)):
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
        groups.append(GroupPlan(root=root, name=module_path(module, root), parameters=list(group_plans.values())))

    for plan in known.values():
        # PyTorch accepts a gradient of any dtype with grad_dtype None, but a gradient buffer needs one dtype.
        if plan.trainable and plan.sharded.grad_dtype is None:
            raise ValueError(
                f"The trainable parameter ({plan.name}) has grad_dtype None, so the runtime cannot allocate its "
                "gradient buffer. Set grad_dtype to a dtype, e.g. param.grad_dtype = param.dtype."
            )

    return groups
