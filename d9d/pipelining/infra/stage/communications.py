import dataclasses
from typing import Any, Generic, cast

import torch
import torch.distributed as dist

from d9d.core import pytree
from d9d.core.types import PyTree, TensorSpec
from d9d.pipelining.api import TStageTransfer


def _is_spec(node: Any) -> bool:
    return isinstance(node, TensorSpec)


@dataclasses.dataclass(frozen=True, slots=True)
class _MicrobatchReceivePlan:
    """How to receive and rebuild one microbatch's incoming transfer.

    Attributes:
        leaf_specs: The specs of the ordered tensor leaves to receive, in transfer flatten order;
            each sizes one receive buffer.
        treespec: The structure spec that rebuilds the ``StageTransfer`` from the received leaves.
    """

    leaf_specs: list[TensorSpec]
    treespec: pytree.PyTreeSpec


def _build_receive_plan(spec_per_microbatch: tuple[PyTree[TensorSpec], ...]) -> list[_MicrobatchReceivePlan]:
    plan_per_microbatch: list[_MicrobatchReceivePlan] = []

    for pytree_specs in spec_per_microbatch:
        # tree_flatten is guaranteed to be deterministic in order
        leaf_specs, treespec = pytree.tree_flatten(pytree_specs, is_leaf=_is_spec)
        plan_per_microbatch.append(_MicrobatchReceivePlan(leaf_specs=leaf_specs, treespec=treespec))

    return plan_per_microbatch


class StageReceiver(Generic[TStageTransfer]):
    """Receives one stage's incoming ``StageTransfer`` from a single peer stage."""

    def __init__(
        self,
        peer_global_rank: int,
        spec_per_microbatch: tuple[PyTree[TensorSpec], ...],
        group: dist.ProcessGroup,
        requires_grad: bool,
    ):
        """Constructs a StageReceiver object.

        Args:
            peer_global_rank: The global (world) rank of the peer stage sending the transfer.
            spec_per_microbatch: The incoming transfer spec (a ``TensorSpec`` PyTree) for each
                microbatch in the pack; the receive buffers for microbatch ``i`` are sized from
                entry ``i``.
            group: The process group strictly for pipeline communication.
            requires_grad: Whether receive buffers should require gradients (enables gradient flow
                from backward stages to forward stages).
        """
        self._peer_global_rank = peer_global_rank
        self._group = group
        self._requires_grad = requires_grad
        self._plan_per_microbatch = _build_receive_plan(spec_per_microbatch)
        self._live_buffers: dict[int, list[torch.Tensor]] = {}

    def _allocate_buffer(self, spec: TensorSpec) -> torch.Tensor:
        return torch.empty(
            spec.shape,
            dtype=spec.dtype,
            layout=spec.layout,
            device="cuda",  # force device
            requires_grad=self._requires_grad,
        )

    def set_inputs_local(self, inputs: TStageTransfer, microbatch_index: int):
        """Manually fills the input buffer for a specific microbatch with a local transfer.

        Used for the V-shape schedulers, where the producing stage lives on the same rank and its
        transfer is handed over directly rather than received via the network.

        Args:
            inputs: The ``StageTransfer`` produced by the peer stage.
            microbatch_index: The microbatch identifier.
        """
        self._live_buffers[microbatch_index] = [
            leaf.detach().requires_grad_(self._requires_grad) for leaf in pytree.tree_leaves(inputs)
        ]

    def pop_inputs(self, microbatch_index: int) -> TStageTransfer:
        """Retrieves and releases the input transfer for a specific microbatch.

        Consume-once: the buffers are removed from the handler, transferring ownership to the caller so
        the memory can be freed once the caller (e.g. the forward/backward cache) releases it. Calling
        this twice for the same microbatch raises ``KeyError``.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The received ``StageTransfer``, reconstructed from the buffers.

        Raises:
            KeyError: If no buffer has been allocated for the microbatch (never received or set).
        """
        treespec = self._plan_per_microbatch[microbatch_index].treespec
        buffers = self._live_buffers.pop(microbatch_index)
        return cast(TStageTransfer, pytree.tree_unflatten(treespec, buffers))

    def receive(self, microbatch_index: int) -> list[dist.P2POp]:
        """Allocates the receive buffers for a microbatch and generates the P2P receive operations.

        Allocates one buffer per transfer leaf (in flatten order) and registers them as live until
        consumed by :meth:`pop_inputs`, then builds the ``dist.irecv`` ops that fill them.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            A list of `dist.P2POp` objects configured for `dist.irecv`.
        """
        ops = []
        buffers = []

        for leaf_spec in self._plan_per_microbatch[microbatch_index].leaf_specs:
            buffer = self._allocate_buffer(leaf_spec)
            buffers.append(buffer)
            ops.append(dist.P2POp(dist.irecv, buffer, self._peer_global_rank, self._group))

        self._live_buffers[microbatch_index] = buffers
        return ops

    def reset(self):
        """Resets the internal state, releasing any live receive buffers."""
        self._live_buffers.clear()


class StageSender(Generic[TStageTransfer]):
    """Sends one stage's outgoing ``StageTransfer`` to a single peer stage."""

    def __init__(self, peer_global_rank: int, group: dist.ProcessGroup):
        """Constructs a StageSender object.

        Args:
            peer_global_rank: The global (world) rank of the peer stage consuming the transfer.
            group: The process group strictly for pipeline communication.
        """
        self._peer_global_rank = peer_global_rank
        self._group = group

    def send(self, send_contents: TStageTransfer) -> list[dist.P2POp]:
        """Generates the PyTorch P2P send operations for a transfer.

        Args:
            send_contents: The ``StageTransfer`` to send (only its tensor leaves are read).

        Returns:
            A list of `dist.P2POp` objects configured for `dist.isend`.
        """
        return [
            dist.P2POp(dist.isend, leaf, self._peer_global_rank, self._group)
            for leaf in pytree.tree_leaves(send_contents)
        ]
