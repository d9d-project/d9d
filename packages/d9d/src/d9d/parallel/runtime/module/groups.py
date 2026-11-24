import dataclasses
from collections.abc import Generator, Sequence
from contextlib import contextmanager

import torch
from torch import nn

from ..bucket import ParameterHandle, WeightGather, bind_parameters, install_tensor_hook_bridge, release_parameter
from ..plan import module_path


@dataclasses.dataclass(slots=True)
class BoundGroup:
    """One parameter group of a bound module.

    Attributes:
        root: The root module of the group.
        name: The name of the group, which errors report.
        handles: The parameters of the group.
        gathers: The weight gathers of the group.
    """

    root: nn.Module
    name: str
    handles: list[ParameterHandle]
    gathers: list[WeightGather]


class ParameterGroups:
    """The parameter groups of one bound module, which put the unsharded weights of a group into its modules."""

    def __init__(self, module: nn.Module, groups: list[BoundGroup]):
        """Constructs the ``ParameterGroups`` object.

        It makes the tensor hooks of the sharded parameters run on the gradients of the unsharded ones.

        Args:
            module: The registered module, which errors name the other modules from.
            groups: The groups of the module.
        """
        self._module = module
        self._name = type(module).__name__
        self._groups = groups
        self._indices = {group.root: index for index, group in enumerate(groups)}
        self._closed = False
        for handle in self._handles():
            install_tensor_hook_bridge(handle)

    def _handles(self) -> list[ParameterHandle]:
        # A tied parameter is in several groups.
        return list(dict.fromkeys(handle for group in self._groups for handle in group.handles))

    def require_open(self):
        """Checks that the groups are not closed.

        Raises:
            ValueError: If ``close()`` ran.
        """
        if self._closed:
            raise ValueError(
                f"The parameters of the module ({self._name}) belong to a closed bind. Call parameters() on the "
                "BoundParameters of a new bind()."
            )

    def index(self, root: nn.Module) -> int:
        """Returns the index of the group of a root.

        Args:
            root: The root of the group.

        Returns:
            The index of the group.

        Raises:
            ValueError: If ``root`` is not the root of a group.
        """
        index = self._indices.get(root)
        if index is None:
            raise ValueError(
                f"The module ({module_path(self._module, root)}) is not the root of a parameter group of the module "
                f"({self._name}). Pass it in the group roots of register()."
            )
        return index

    def root(self, index: int) -> nn.Module:
        """Returns the root of a group.

        Args:
            index: The index of the group.

        Returns:
            The root of the group.
        """
        return self._groups[index].root

    def name(self, index: int) -> str:
        """Returns the name of a group.

        Args:
            index: The index of the group.

        Returns:
            The name of the group.
        """
        return self._groups[index].name

    def omitted(self, order: Sequence[int | None]) -> list[str]:
        """Returns the names of the parameters that no group of an order holds.

        Args:
            order: The index of the group of each region, or ``None`` for a region without a group.

        Returns:
            The names of the parameters.
        """
        used = {handle for index in order if index is not None for handle in self._groups[index].handles}
        return [handle.plan.name for handle in self._handles() if handle not in used]

    @contextmanager
    def weights(self, index: int, next_index: int | None) -> Generator[None, None, None]:
        """Puts the unsharded weights of a group into its modules, as ``ModuleParameters.group_weights()`` does.

        Args:
            index: The index of the group.
            next_index: The index of the group of the next region, or ``None``.
        """
        # A fresh event for every region: the sharded weights can change between regions, e.g. by an optimizer step
        # before an evaluation. In steady state, the gather stream already waits for the previous region, so the event
        # adds no wait.
        weights_written = torch.cuda.current_stream().record_event()
        group = self._groups[index]
        for gather in group.gathers:
            gather.start(weights_written)
            gather.finish()
        # The gathers of this group start first, so the prefetch does not delay them.
        following = [] if next_index is None else self._groups[next_index].gathers
        for gather in following:
            gather.start(weights_written)

        bind_parameters(group.handles, unsharded=True)
        try:
            yield
        finally:
            bind_parameters(group.handles, unsharded=False)
            for gather in group.gathers:
                # A gather that the next group uses too stays, otherwise that group gathers it again.
                if gather not in following:
                    gather.free()

    def close(self):
        """Removes the hooks and frees the unsharded parameters.

        Afterwards, ``require_open()`` raises.
        """
        for handle in self._handles():
            release_parameter(handle)
        self._closed = True
        self._indices = {}
        self._groups = []
