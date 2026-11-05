from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Generic, cast

import torch
from torch import nn

from d9d import pytree
from d9d.autograd.split_backward import (
    PendingWeightBackward,
    stage_backward_full,
    stage_backward_input,
    stage_backward_weight,
)
from d9d.pipelining.api import TPipelineInput, TStageTransfer

from ..windows import WindowChain, backward_chains

if TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime


@dataclasses.dataclass(kw_only=True, slots=True)
class _SendableInputGrads:
    """Input gradients ready to send back to the previous stage.

    Attributes:
        leaves: The input gradient tensor leaves, in the stage's input-transfer flatten order.
        treespec: The structure spec that rebuilds the input ``StageTransfer`` from ``leaves``.
    """

    leaves: list[torch.Tensor | None]
    treespec: pytree.PyTreeSpec


@dataclasses.dataclass(kw_only=True, slots=True)
class _WeightPhaseFullBackward:
    """A whole backward pass that the weight phase runs.

    The first stage uses it. It has no previous stage to send input gradients to, so it does not split
    the backward pass. It keeps the flattened tensors and runs a full backward in the weight phase.

    Attributes:
        outputs: The output tensor leaves to backprop from.
        output_grads: Their gradient leaves, or ``None`` to seed an implicit unit gradient.
        inputs: The input tensor leaves the backward pass flows into.
        chain: The executed regions, or ``None`` if the stage ran as one autograd graph.
    """

    outputs: list[torch.Tensor]
    output_grads: list[torch.Tensor] | None
    inputs: list[torch.Tensor]
    chain: WindowChain | None = None


@dataclasses.dataclass(kw_only=True, slots=True)
class _WeightPhaseWindows:
    """The weight backward of a region chain, which keeps its own pending work.

    Attributes:
        chain: The executed regions.
    """

    chain: WindowChain


@dataclasses.dataclass(kw_only=True, slots=True)
class _BackwardState:
    """Per-microbatch backward state carried between the backward phases.

    Each field can be ``None`` independently of the other.

    Attributes:
        sendable_grads: The input gradients ready to send upstream, or ``None`` on the first stage
            and after they were sent.
        pending_weight: The weight-gradient work still to run, or ``None`` after a full backward.
    """

    sendable_grads: _SendableInputGrads | None
    pending_weight: _WeightPhaseFullBackward | PendingWeightBackward | _WeightPhaseWindows | None


@dataclasses.dataclass(slots=True)
class BackwardSeedLoss:
    """Backward seed for the last stage.

    The produced output does not leave the pipeline, so the backward pass is seeded directly from the scalar
    loss with an implicit unit gradient.

    Attributes:
        loss: The scalar loss produced by the last stage.
    """

    loss: torch.Tensor


@dataclasses.dataclass(slots=True)
class BackwardSeedTransfer(Generic[TStageTransfer]):
    """Backward seed for a non-last stage.

    The backward pass propagates the gradients received from the next stage through the transfer this stage
    produced in the forward pass.

    Attributes:
        outputs: The ``StageTransfer`` this stage produced in the forward pass.
        output_grads: The gradients of the loss w.r.t. ``outputs``, received from the next stage.
    """

    outputs: TStageTransfer
    output_grads: TStageTransfer


BackwardSeed = BackwardSeedLoss | BackwardSeedTransfer[TStageTransfer]


class BackwardComputeHandler(Generic[TPipelineInput, TStageTransfer]):
    """Backward pass handler for a pipeline stage.

    It can split the backward pass into an input-gradient and a weight-gradient phase, as Zero Bubble
    schedules require.
    """

    def __init__(
        self, stage_index: int, module: nn.Module, has_input_peer: bool, runtime: ParameterRuntime | None = None
    ):
        """Constructs the ``BackwardComputeHandler`` object.

        Args:
            stage_index: Logical index of the stage.
            module: The PyTorch module to compute gradients for.
            has_input_peer: Whether this stage has a previous stage to send input gradients to.
            runtime: The ``ParameterRuntime`` that owns the unsharded parameters, or ``None`` if the stage runs as
                one autograd graph.
        """
        self._stage_idx = stage_index
        self._module = module
        self._has_input_peer = has_input_peer
        self._runtime = runtime

        self._cache: dict[int, _BackwardState] = {}

    def _parameters_with_grad(self) -> Iterator[nn.Parameter]:
        return (param for param in self._module.parameters() if param.requires_grad)

    def _release_if_complete(self, microbatch_index: int):
        state = self._cache[microbatch_index]
        if state.sendable_grads is None and state.pending_weight is None:
            del self._cache[microbatch_index]

    @staticmethod
    def _seed_leaves(seed: BackwardSeed[TStageTransfer]) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        match seed:
            case BackwardSeedLoss():
                return [seed.loss], None
            case BackwardSeedTransfer():
                return pytree.tree_leaves(seed.outputs), pytree.tree_leaves(seed.output_grads)
            case _:
                raise ValueError(f"Unknown backward seed type ({type(seed).__name__}).")

    def backward_full(
        self,
        microbatch_index: int,
        inputs: TPipelineInput | TStageTransfer,
        seed: BackwardSeed[TStageTransfer],
        chain: WindowChain | None = None,
        final: bool | None = None,
    ):
        """Performs a full backward pass (both inputs and weights).

        Args:
            microbatch_index: Identifier for the microbatch.
            inputs: The input transfer used in the forward pass.
            seed: The downstream backward seed: the loss on the last stage, the transfer and its
                gradients otherwise.
            chain: The executed regions, or ``None`` if the stage ran as one autograd graph.
            final: Whether this pass is the last one of the stage that writes weight gradients in the step, or
                ``None`` to skip the reductions.

        Raises:
            ValueError: If a backward pass already ran for this microbatch.
        """
        if microbatch_index in self._cache:
            raise ValueError(
                f"A backward pass already ran on stage ({self._stage_idx}) for microbatch_index ({microbatch_index})."
            )

        input_leaves, input_spec = pytree.tree_flatten(inputs)
        output_leaves, output_grad_leaves = self._seed_leaves(seed)

        if chain is None:
            inputs_grad_linear = stage_backward_full(
                outputs=output_leaves,
                output_grads=output_grad_leaves,
                inputs=input_leaves,
            )
        else:
            chain.start_backward(output_leaves, output_grad_leaves)
            # The caller already holds the compute context of this microbatch.
            backward_chains([chain], self._require_runtime(), self._stage_idx, final, [nullcontext()])
            inputs_grad_linear = chain.input_grads(input_leaves)

        sendable = _SendableInputGrads(leaves=inputs_grad_linear, treespec=input_spec) if self._has_input_peer else None
        self._cache[microbatch_index] = _BackwardState(sendable_grads=sendable, pending_weight=None)
        self._release_if_complete(microbatch_index)

    def backward_window(
        self,
        seeds: Sequence[BackwardSeed[TStageTransfer]],
        chains: Sequence[WindowChain],
        final: bool | None,
        contexts: Sequence[AbstractContextManager[None]],
    ):
        """Performs a full backward pass of several microbatches of a stage without peers, region by region.

        Each region gets its weights once for all of them and reduces once, after the last microbatch. A stage
        without peers sends no input gradients and defers no weight gradients, so nothing is cached.

        Args:
            seeds: The backward seed of each microbatch.
            chains: The executed regions of each microbatch.
            final: Whether this pass is the last one of the stage that writes weight gradients in the step, or
                ``None`` to skip the reductions.
            contexts: The compute context of each microbatch. Each is entered once per region.
        """
        for chain, seed in zip(chains, seeds, strict=True):
            chain.start_backward(*self._seed_leaves(seed))
        backward_chains(chains, self._require_runtime(), self._stage_idx, final, contexts)

    def _require_runtime(self) -> ParameterRuntime:
        if self._runtime is None:
            raise ValueError(f"Stage ({self._stage_idx}) has no ParameterRuntime, so it cannot run region chains.")
        return self._runtime

    def backward_input(
        self,
        microbatch_index: int,
        inputs: TPipelineInput | TStageTransfer,
        seed: BackwardSeed[TStageTransfer],
        chain: WindowChain | None = None,
    ):
        """Performs a partial backward pass to compute gradients with respect to inputs only.

        A later ``backward_weight`` call for the same microbatch computes the weight gradients.

        Args:
            microbatch_index: Identifier for the microbatch.
            inputs: The input transfer used in the forward pass.
            seed: The downstream backward seed: the loss on the last stage, the transfer and its
                gradients otherwise.
            chain: The executed regions, or ``None`` if the stage ran as one autograd graph.

        Raises:
            ValueError: If a backward pass already ran for this microbatch.
        """
        if microbatch_index in self._cache:
            raise ValueError(
                f"A backward pass already ran on stage ({self._stage_idx}) for microbatch_index ({microbatch_index})."
            )

        input_leaves, input_spec = pytree.tree_flatten(inputs)
        output_leaves, output_grad_leaves = self._seed_leaves(seed)

        if self._has_input_peer and chain is not None:
            chain.start_backward(output_leaves, output_grad_leaves)
            chain.backward_input()
            self._cache[microbatch_index] = _BackwardState(
                sendable_grads=_SendableInputGrads(leaves=chain.input_grads(input_leaves), treespec=input_spec),
                pending_weight=_WeightPhaseWindows(chain=chain),
            )
        elif self._has_input_peer:
            input_grads, pending_weight = stage_backward_input(
                outputs=output_leaves,
                output_grads=output_grad_leaves,
                inputs=input_leaves,
                weights=self._parameters_with_grad(),
            )

            self._cache[microbatch_index] = _BackwardState(
                sendable_grads=_SendableInputGrads(leaves=input_grads, treespec=input_spec),
                pending_weight=pending_weight,
            )
        else:
            self._cache[microbatch_index] = _BackwardState(
                sendable_grads=None,
                pending_weight=_WeightPhaseFullBackward(
                    outputs=output_leaves,
                    output_grads=output_grad_leaves,
                    inputs=input_leaves,
                    chain=chain,
                ),
            )

    def backward_weight(self, microbatch_index: int, final: bool | None = None):
        """Performs a partial backward pass to accumulate gradients into weights.

        ``backward_input`` must run first for the same microbatch.

        Args:
            microbatch_index: Identifier for the microbatch.
            final: Whether this pass is the last one of the stage that writes weight gradients in the step, or
                ``None`` to skip the reductions.

        Raises:
            ValueError: If ``backward_input`` did not run for this microbatch, or if the weight
                backward already ran.
        """
        if microbatch_index not in self._cache:
            raise ValueError(
                f"Weight backward on stage ({self._stage_idx}) for microbatch_index ({microbatch_index}) "
                "needs a preceding input backward."
            )

        state = self._cache[microbatch_index]
        pending = state.pending_weight

        match pending:
            case _WeightPhaseFullBackward():
                if pending.chain is None:
                    stage_backward_full(
                        outputs=pending.outputs,
                        output_grads=pending.output_grads,
                        inputs=pending.inputs,
                    )
                else:
                    pending.chain.start_backward(pending.outputs, pending.output_grads)
                    # The caller already holds the compute context of this microbatch.
                    backward_chains([pending.chain], self._require_runtime(), self._stage_idx, final, [nullcontext()])
            case _WeightPhaseWindows():
                pending.chain.backward_weight(final)
            case PendingWeightBackward():
                stage_backward_weight(pending)
            case None:
                raise ValueError(
                    f"No weight backward is pending on stage ({self._stage_idx}) for microbatch_index "
                    f"({microbatch_index}): the previous backward pass was a full backward, or the weight backward "
                    "already ran."
                )
            case _:
                raise ValueError(f"Unknown pending weight backward type ({type(pending)}).")

        state.pending_weight = None
        self._release_if_complete(microbatch_index)

    def pop_for_sending(self, microbatch_index: int) -> TStageTransfer:
        """Retrieves the calculated input gradients for a microbatch as a transfer PyTree.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            The input gradients, shaped like the stage's input transfer.

        Raises:
            ValueError: If no input gradients are ready for sending, or if an input gradient is
                ``None``.
        """
        state = self._cache[microbatch_index]

        sendable = state.sendable_grads
        if sendable is None:
            raise ValueError(
                f"No input gradients to send on stage ({self._stage_idx}) for microbatch_index ({microbatch_index}). "
                "Call backward_full() or backward_input() first."
            )

        for grad_value in sendable.leaves:
            if grad_value is None:
                raise ValueError(
                    f"An input gradient on stage ({self._stage_idx}) for microbatch_index ({microbatch_index}) "
                    "is None and cannot be sent. Every tensor in the incoming StageTransfer must have a "
                    "differentiable path to the stage outputs."
                )

        full_tree = pytree.tree_unflatten(sendable.treespec, sendable.leaves)

        state.sendable_grads = None
        self._release_if_complete(microbatch_index)

        return cast(TStageTransfer, full_tree)
