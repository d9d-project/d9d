import dataclasses
from collections.abc import Iterator
from typing import Generic, cast

import torch
from torch import nn
from torch.autograd.graph import Node

from d9d.core import pytree
from d9d.pipelining.api import TPipelineInput, TPipelineOutput, TSharedInput, TStageTransfer

from .splitgrad import (
    ParamGroup,
    stage_backward_full,
    stage_backward_input,
    stage_backward_weight,
)

# TODO/NOTICE: We WILL NOT disable FSDP's resharding for microbatches since it will modify
# TODO/NOTICE: its behavior in an unexpected way. Perhaps we need better FSDP resharding policy handler?


@dataclasses.dataclass(slots=True)
class ForwardCache(Generic[TPipelineInput, TStageTransfer, TPipelineOutput]):
    """Stores the inputs and outputs of a forward pass to be used later in the backward pass.

    Attributes:
        inputs: The stage's incoming ``StageTransfer`` (or ``PipelineInput`` on the first stage) as a
            PyTree.
        outputs: The stage's produced ``StageTransfer`` (or ``PipelineOutput`` on the last stage) as a
            PyTree.
    """

    inputs: TPipelineInput | TStageTransfer
    outputs: TStageTransfer | TPipelineOutput


class ForwardComputeHandler(Generic[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
    """Handles the execution of the forward pass for a pipeline stage module.

    Maintains a cache of inputs and outputs indexed by microbatch ID.
    """

    def __init__(self, stage_index: int, module: nn.Module):
        """Constructs a ForwardComputeHandler object.

        Args:
            stage_index: Logical index of the stage.
            module: The PyTorch module representing this stage computation.
        """
        self._stage_idx = stage_index
        self._module = module

        self._cache: dict[int, ForwardCache[TPipelineInput, TStageTransfer, TPipelineOutput]] = {}

    def run(self, microbatch_index: int, inputs: TPipelineInput | TStageTransfer, shared: TSharedInput):
        """Executes the module's forward pass.

        Args:
            microbatch_index: Identifier for the current microbatch.
            inputs: The stage's ``PipelineInput`` (first stage) or incoming ``StageTransfer``.
            shared: The ``SharedInput`` passed to every stage.

        Raises:
            RuntimeError: If the forward pass implementation fails.
        """
        try:
            output = self._module(inputs, shared)
        except Exception as e:
            raise RuntimeError(f"S{self._stage_idx}B{microbatch_index} failed to run forward") from e

        self._cache[microbatch_index] = ForwardCache(inputs=inputs, outputs=output)

    def get_outputs(self, microbatch_index: int) -> TStageTransfer | TPipelineOutput:
        """Retrieves cached outputs for a specific microbatch without removing them.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            The produced transfer/output PyTree.
        """
        return self._cache[microbatch_index].outputs

    def pop_inputs_outputs(
        self, microbatch_index: int
    ) -> tuple[TPipelineInput | TStageTransfer, TStageTransfer | TPipelineOutput]:
        """Retrieves and removes the cached inputs and outputs for a specific microbatch.

        Typically called when initiating the backward pass.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            A tuple containing (inputs, outputs).
        """
        cache = self._cache.pop(microbatch_index)
        return cache.inputs, cache.outputs


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
class _DeferredFullBackward:
    """A full backward to replay at weight time.

    Used by the first stage, which cannot split input- and weight-gradient passes (it has no input
    peer to send input grads to), so it stashes the flattened tensors and runs a full backward when
    the weight phase arrives.

    Attributes:
        outputs: The output tensor leaves to backprop from.
        output_grads: Their gradient leaves, or None to seed an implicit unit gradient.
        inputs: The input tensor leaves the backward flows into.
    """

    outputs: list[torch.Tensor]
    output_grads: list[torch.Tensor] | None
    inputs: list[torch.Tensor]


@dataclasses.dataclass(kw_only=True, slots=True)
class _DeferredWeightBackward:
    """A weight-gradient-only backward to replay from saved param groups (the ZB split optimization).

    Attributes:
        param_groups: The parameter groups whose weight gradients still need accumulating.
        ownership_tokens: Autograd nodes kept alive to own the pending gradient graph.
    """

    param_groups: list[ParamGroup]
    ownership_tokens: list[Node]


@dataclasses.dataclass(kw_only=True, slots=True)
class _BackwardState:
    """Per-microbatch backward state carried between the backward phases.

    A single entry holds two orthogonal, independently-optional facts:

    Attributes:
        sendable_grads: The input gradients ready to send upstream, or None on the first stage.
        pending_weight: The weight-gradient work still to run, or None after a full backward.
    """

    sendable_grads: _SendableInputGrads | None
    pending_weight: _DeferredFullBackward | _DeferredWeightBackward | None


@dataclasses.dataclass(slots=True)
class BackwardSeedLoss:
    """Backward seed for the last stage.

    The produced output does not leave the pipeline, so backward is seeded directly from the scalar
    loss with an implicit unit gradient.

    Attributes:
        loss: The scalar loss produced by the last stage.
    """

    loss: torch.Tensor


@dataclasses.dataclass(slots=True)
class BackwardSeedTransfer(Generic[TStageTransfer]):
    """Backward seed for a non-last stage.

    Backward propagates the gradients received from the next stage through the transfer this stage
    produced in the forward pass.

    Attributes:
        outputs: The ``StageTransfer`` this stage produced in the forward pass.
        output_grads: The gradients of the loss w.r.t. ``outputs``, received from the next stage.
    """

    outputs: TStageTransfer
    output_grads: TStageTransfer


BackwardSeed = BackwardSeedLoss | BackwardSeedTransfer[TStageTransfer]


class BackwardComputeHandler(Generic[TPipelineInput, TStageTransfer]):
    """Handles the execution of backward passes for a pipeline stage.

    Supports splitting the backward pass into input-gradients and weight-gradients phases, which is
    necessary for schedules like ZB.
    """

    def __init__(self, stage_index: int, module: nn.Module, has_input_peer: bool):
        """Constructs a BackwardComputeHandler object.

        Args:
            stage_index: Logical index of the stage (used only in diagnostics).
            module: The PyTorch module to compute gradients for.
            has_input_peer: Whether this stage has a previous stage to send input gradients to.
        """
        self._stage_idx = stage_index
        self._module = module
        self._has_input_peer = has_input_peer

        self._cache: dict[int, _BackwardState] = {}

    def _parameters_with_grad(self) -> Iterator[nn.Parameter]:
        return (param for param in self._module.parameters() if param.requires_grad)

    def _release_if_complete(self, microbatch_index: int):
        state = self._cache[microbatch_index]
        if state.sendable_grads is None and state.pending_weight is None:
            del self._cache[microbatch_index]

    @staticmethod
    def _seed_leaves(seed: "BackwardSeed[TStageTransfer]") -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        match seed:
            case BackwardSeedLoss():
                return [seed.loss], None
            case BackwardSeedTransfer():
                return pytree.tree_leaves(seed.outputs), pytree.tree_leaves(seed.output_grads)
            case _:
                raise ValueError("Unknown backward seed type")

    def backward_full(
        self,
        microbatch_index: int,
        inputs: TPipelineInput | TStageTransfer,
        seed: "BackwardSeed[TStageTransfer]",
    ):
        """Performs a full backward pass (both inputs and weights).

        Args:
            microbatch_index: Identifier for the microbatch.
            inputs: The input transfer used in the forward pass.
            seed: The downstream backward seed (loss on the last stage, transfer + grads otherwise).

        Raises:
            ValueError: If a double backward is attempted for the same microbatch.
        """
        if microbatch_index in self._cache:
            raise ValueError(f"S{self._stage_idx}B{microbatch_index} double backward")

        input_leaves, input_spec = pytree.tree_flatten(inputs)
        output_leaves, output_grad_leaves = self._seed_leaves(seed)

        inputs_grad_linear = stage_backward_full(
            outputs=output_leaves,
            output_grads=output_grad_leaves,
            inputs=input_leaves,
        )

        sendable = _SendableInputGrads(leaves=inputs_grad_linear, treespec=input_spec) if self._has_input_peer else None
        self._cache[microbatch_index] = _BackwardState(sendable_grads=sendable, pending_weight=None)
        self._release_if_complete(microbatch_index)

    def backward_input(
        self,
        microbatch_index: int,
        inputs: TPipelineInput | TStageTransfer,
        seed: "BackwardSeed[TStageTransfer]",
    ):
        """Performs a partial backward pass to compute gradients with respect to inputs only.

        This prepares the computation state for a subsequent `backward_weight` call.

        Args:
            microbatch_index: Identifier for the microbatch.
            inputs: The input transfer used in the forward pass.
            seed: The downstream backward seed (loss on the last stage, transfer + grads otherwise).

        Raises:
            ValueError: If a double backward is attempted.
        """
        if microbatch_index in self._cache:
            raise ValueError(f"S{self._stage_idx}B{microbatch_index} double backward")

        input_leaves, input_spec = pytree.tree_flatten(inputs)
        output_leaves, output_grad_leaves = self._seed_leaves(seed)

        if self._has_input_peer:
            results = stage_backward_input(
                outputs=output_leaves,
                output_grads=output_grad_leaves,
                inputs=input_leaves,
                weights=self._parameters_with_grad(),
            )

            self._cache[microbatch_index] = _BackwardState(
                sendable_grads=_SendableInputGrads(leaves=results.input_grads, treespec=input_spec),
                pending_weight=_DeferredWeightBackward(
                    param_groups=results.param_groups,
                    ownership_tokens=results.grad_ownership_tokens,
                ),
            )
        else:
            self._cache[microbatch_index] = _BackwardState(
                sendable_grads=None,
                pending_weight=_DeferredFullBackward(
                    outputs=output_leaves,
                    output_grads=output_grad_leaves,
                    inputs=input_leaves,
                ),
            )

    def backward_weight(self, microbatch_index: int):
        """Performs a partial backward pass to accumulate gradients into weights.

        Must be preceded by `backward_input` for the same microbatch index.

        Args:
            microbatch_index: Identifier for the microbatch.

        Raises:
            ValueError: If `backward_input` was not called before or if called twice.
        """
        if microbatch_index not in self._cache:
            raise ValueError(f"S{self._stage_idx}BW{microbatch_index} - weight backward with no input backward before")

        state = self._cache[microbatch_index]
        pending = state.pending_weight

        match pending:
            case _DeferredFullBackward():
                stage_backward_full(
                    outputs=pending.outputs,
                    output_grads=pending.output_grads,
                    inputs=pending.inputs,
                )
            case _DeferredWeightBackward():
                stage_backward_weight(weights=self._parameters_with_grad(), param_groups=pending.param_groups)
            case None:
                raise ValueError("Previous backward was a full backward, not an input backward")

        state.pending_weight = None
        self._release_if_complete(microbatch_index)

    def pop_for_sending(self, microbatch_index: int) -> TStageTransfer:
        """Retrieves the calculated input gradients for a microbatch as a transfer PyTree.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            The input gradients, shaped like the stage's input transfer.

        Raises:
            ValueError: If the required backward pass was not performed or if gradients are missing.
        """
        state = self._cache[microbatch_index]

        sendable = state.sendable_grads
        if sendable is None:
            raise ValueError("You should call either backward_full or backward_input before popping cached grad")

        for grad_value in sendable.leaves:
            if grad_value is None:
                raise ValueError("Cannot pop null gradient for sending! Perhaps malformed schedule?")

        full_tree = pytree.tree_unflatten(sendable.treespec, sendable.leaves)

        state.sendable_grads = None
        self._release_if_complete(microbatch_index)

        return cast(TStageTransfer, full_tree)
