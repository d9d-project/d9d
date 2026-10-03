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
        leaf_specs: The specs of the tensor leaves to receive, in transfer flatten order. Each spec
            sizes one receive buffer.
        treespec: The structure spec that rebuilds the ``StageTransfer`` from the received leaves.
    """

    leaf_specs: list[TensorSpec]
    treespec: pytree.PyTreeSpec


def _build_receive_plan(spec_per_microbatch: tuple[PyTree[TensorSpec], ...]) -> list[_MicrobatchReceivePlan]:
    plan_per_microbatch: list[_MicrobatchReceivePlan] = []

    for pytree_specs in spec_per_microbatch:
        # Sender and receiver match leaves by flatten order, and tree_flatten keeps that order deterministic.
        leaf_specs, treespec = pytree.tree_flatten(pytree_specs, is_leaf=_is_spec)
        plan_per_microbatch.append(_MicrobatchReceivePlan(leaf_specs=leaf_specs, treespec=treespec))

    return plan_per_microbatch


class StageReceiver(Generic[TStageTransfer]):
    """Receiver of one stage's incoming ``StageTransfer`` from a single peer stage."""

    def __init__(
        self,
        peer_global_rank: int,
        spec_per_microbatch: tuple[PyTree[TensorSpec], ...],
        group: dist.ProcessGroup,
        requires_grad: bool,
    ):
        """Constructs the ``StageReceiver`` object.

        Args:
            peer_global_rank: The global (world) rank of the peer stage sending the transfer.
            spec_per_microbatch: The incoming transfer spec (a ``TensorSpec`` PyTree) for each
                microbatch in the pack. Entry ``i`` sizes the receive buffers of microbatch ``i``.
            group: The pipeline-parallel process group.
            requires_grad: Whether receive buffers require gradients, so that the backward pass
                can compute gradients for them.
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
            # TensorSpec has no device field: receive buffers always live on the current CUDA device.
            device="cuda",
            requires_grad=self._requires_grad,
        )

    def set_inputs_local(self, inputs: TStageTransfer, microbatch_index: int):
        """Fills the input buffers of a microbatch with a transfer from a stage on the same rank.

        V-shape schedules use it when the producing stage lives on the same rank, so the transfer
        does not go over the network.

        Args:
            inputs: The ``StageTransfer`` produced by the peer stage.
            microbatch_index: The microbatch identifier.
        """
        self._live_buffers[microbatch_index] = [
            leaf.detach().requires_grad_(self._requires_grad) for leaf in pytree.tree_leaves(inputs)
        ]

    def pop_inputs(self, microbatch_index: int) -> TStageTransfer:
        """Retrieves and releases the input transfer for a specific microbatch.

        The buffers are removed from the receiver, so the caller owns them and can free them. A second
        call for the same microbatch raises ``KeyError``.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The received ``StageTransfer``, reconstructed from the buffers.

        Raises:
            KeyError: If the microbatch has no live buffers: they were never received or set, or were
                already popped.
        """
        treespec = self._plan_per_microbatch[microbatch_index].treespec
        buffers = self._live_buffers.pop(microbatch_index)
        return cast(TStageTransfer, pytree.tree_unflatten(treespec, buffers))

    def receive(self, microbatch_index: int) -> list[dist.P2POp]:
        """Allocates the receive buffers for a microbatch and builds the P2P receive operations.

        The buffers stay live until ``pop_inputs`` consumes them.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            A list of ``dist.P2POp`` objects for ``dist.irecv``.
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
    """Sender of one stage's outgoing ``StageTransfer`` to a single peer stage."""

    def __init__(self, peer_global_rank: int, group: dist.ProcessGroup):
        """Constructs the ``StageSender`` object.

        Args:
            peer_global_rank: The global (world) rank of the peer stage consuming the transfer.
            group: The pipeline-parallel process group.
        """
        self._peer_global_rank = peer_global_rank
        self._group = group

    def send(self, send_contents: TStageTransfer) -> list[dist.P2POp]:
        """Builds the P2P send operations for a transfer.

        Args:
            send_contents: The ``StageTransfer`` to send (only its tensor leaves are read).

        Returns:
            A list of ``dist.P2POp`` objects for ``dist.isend``.
        """
        return [
            dist.P2POp(dist.isend, leaf, self._peer_global_rank, self._group)
            for leaf in pytree.tree_leaves(send_contents)
        ]
