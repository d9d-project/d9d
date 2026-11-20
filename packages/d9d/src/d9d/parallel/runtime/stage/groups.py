"""The parameter groups of a stage, which put the unsharded weights of a group into its modules."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager

import torch
from torch import nn

from ..bucket import ParameterHandle, WeightGather, bind_parameters, install_tensor_hook_bridge, release_parameter


class ParameterGroups:
    """The parameter groups of one bound stage, which put the unsharded weights of a group into its modules."""

    def __init__(
        self,
        stage_index: int,
        roots: Sequence[nn.Module],
        handles_by_group: list[list[ParameterHandle]],
        gathers_by_group: list[list[WeightGather]],
    ):
        """Constructs the ``ParameterGroups`` object.

        It makes the tensor hooks of the sharded parameters run on the gradients of the unsharded ones.

        Args:
            stage_index: The index of the stage.
            roots: The roots of the groups. The first one is the stage root.
            handles_by_group: The parameters that each group holds, by group index.
            gathers_by_group: The weight gathers that each group uses, by group index.
        """
        self._stage_index = stage_index
        self._indices = {root: index for index, root in enumerate(roots)}
        self._handles_by_group = handles_by_group
        self._gathers_by_group = gathers_by_group
        self._closed = False
        for handle in self._handles():
            install_tensor_hook_bridge(handle)

    def _handles(self) -> list[ParameterHandle]:
        # A tied parameter is in several groups.
        return list(dict.fromkeys(handle for group in self._handles_by_group for handle in group))

    def require_open(self):
        """Checks that the runtime is still bound.

        Raises:
            ValueError: If the runtime was unbound since.
        """
        if self._closed:
            raise ValueError(
                f"The runtime was unbound after stage ({self._stage_index}) was taken. Call stage() again after bind()."
            )

    def index(self, module: nn.Module) -> int:
        """Returns the index of the group of a root.

        Args:
            module: The root of the group.

        Returns:
            The index of the group.

        Raises:
            ValueError: If the module is not the root of a group.
        """
        index = self._indices.get(module)
        if index is None:
            raise ValueError(
                f"The module ({type(module).__name__}) is not the root of a parameter group of stage "
                f"({self._stage_index}). Pass it in the groups of register_stage()."
            )
        return index

    def check(self, group: int):
        """Checks that a group index exists.

        Args:
            group: The index of the group.

        Raises:
            ValueError: If the stage has no such group.
        """
        if not 0 <= group < len(self._handles_by_group):
            raise ValueError(
                f"The group ({group}) does not exist in stage ({self._stage_index}), which has "
                f"({len(self._handles_by_group)}) groups. Pass an index from group_index() or from the recorded order."
            )

    def omitted(self, order: Sequence[int | None]) -> list[str]:
        """Returns the names of the parameters that no group of an order uses.

        Args:
            order: The group of each region, or ``None`` for a region without a group.

        Returns:
            The names of the omitted parameters.
        """
        used = {handle for group in order if group is not None for handle in self._handles_by_group[group]}
        return [handle.plan.name for handle in self._handles() if handle not in used]

    @contextmanager
    def weights(self, group: int, next_group: int | None) -> Generator[None, None, None]:
        """Puts the unsharded weights of a group into its modules.

        It gathers the weights that a previous region did not prefetch or keep. On exit, it frees the weights that
        ``next_group`` does not use.

        Args:
            group: The index of the group.
            next_group: The group of the next region, or ``None``. Its weights start gathering before this region.

        Yields:
            Nothing.
        """
        # A fresh event for every region: the sharded weights can change between regions, e.g. by an optimizer step
        # before an evaluation. In steady state, the gather stream already waits for the previous region, so the event
        # adds no wait.
        weights_written = torch.cuda.current_stream().record_event()
        gathers = self._gathers_by_group[group]
        for gather in gathers:
            gather.start(weights_written)
            gather.finish()
        # The gathers of this group start first, so the prefetch does not delay them.
        following = [] if next_group is None else self._gathers_by_group[next_group]
        for gather in following:
            gather.start(weights_written)

        handles = self._handles_by_group[group]
        bind_parameters(handles, unsharded=True)
        try:
            yield
        finally:
            bind_parameters(handles, unsharded=False)
            for gather in gathers:
                # A gather that the next group uses too stays, otherwise that group gathers it again.
                if gather not in following:
                    gather.free()

    def close(self):
        """Removes the hooks and empties the unsharded parameters.

        Afterwards, ``require_open()`` raises.
        """
        for handle in self._handles():
            release_parameter(handle)
        self._closed = True
        self._indices = {}
        self._handles_by_group = []
        self._gathers_by_group = []
