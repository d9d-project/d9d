from typing import Generic, cast

import torch
import torch.distributed as dist
from torch import nn

from d9d.core import pytree
from d9d.pipelining.api import (
    ModuleSupportsPipelining,
    PipelineStageInfo,
    StageBoundary,
    TPipelineInput,
    TPipelineOutput,
    TSharedInput,
    TStageTransfer,
)

from .communications import StageReceiver, StageSender
from .computations import (
    BackwardComputeHandler,
    BackwardSeed,
    BackwardSeedLoss,
    BackwardSeedTransfer,
    ForwardComputeHandler,
)


class PipelineStage(Generic[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
    """A single stage of a pipelined model.

    It combines the P2P handlers (``StageReceiver`` and ``StageSender``) with the compute handlers
    (``ForwardComputeHandler`` and ``BackwardComputeHandler``). It hides buffer management, P2P
    communication and gradient computation from the schedule.
    """

    def __init__(
        self,
        info: PipelineStageInfo,
        module: nn.Module,
        group: dist.ProcessGroup,
        stage_to_host_topology: dict[int, int],
    ):
        """Constructs the ``PipelineStage`` object.

        Args:
            info: The position of the stage in the pipeline.
            module: The module that runs this stage.
            group: The pipeline-parallel process group.
            stage_to_host_topology: Mapping from stage index to the pipeline-parallel rank that hosts it.
        """
        self._info = info
        self._module = module
        self._group = group
        self._stage_to_host_topology = stage_to_host_topology

        self._configured = False

        self._forward_receiver: StageReceiver[TStageTransfer] | None = None
        self._forward_sender: StageSender[TStageTransfer] | None = None
        self._backward_receiver: StageReceiver[TStageTransfer] | None = None
        self._backward_sender: StageSender[TStageTransfer] | None = None

        self._forward_comp: ForwardComputeHandler[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput] = (
            ForwardComputeHandler(stage_index=info.current_stage, module=module)
        )
        self._backward_comp: BackwardComputeHandler[TPipelineInput, TStageTransfer] | None = None

    @property
    def info(self) -> PipelineStageInfo:
        """The position of this stage in the pipeline."""
        return self._info

    def _peer_global_rank(self, stage_idx: int) -> int:
        return dist.get_global_rank(self._group, self._stage_to_host_topology[stage_idx])

    def _make_receiver(
        self,
        inputs: tuple[TPipelineInput, ...],
        module: ModuleSupportsPipelining,
        from_stage: int | None,
        boundary: StageBoundary,
        requires_grad: bool,
    ) -> StageReceiver[TStageTransfer] | None:
        if from_stage is None:
            return None

        return StageReceiver(
            peer_global_rank=self._peer_global_rank(from_stage),
            spec_per_microbatch=tuple(module.stage_transfer_spec(microbatch, boundary) for microbatch in inputs),
            group=self._group,
            requires_grad=requires_grad,
        )

    def _make_sender(self, to_stage: int | None) -> StageSender[TStageTransfer] | None:
        if to_stage is None:
            return None

        return StageSender(peer_global_rank=self._peer_global_rank(to_stage), group=self._group)

    def configure_buffers(self, has_backward: bool, pipeline_inputs_per_microbatch: tuple[TPipelineInput, ...]):
        """Initializes the communication handlers and buffers for the stage.

        It must run before the stage executes any microbatch. Transfer shapes are inferred per
        microbatch, so the microbatches in a pack can differ in shape.

        Args:
            has_backward: Whether the stage must prepare for a backward pass.
            pipeline_inputs_per_microbatch: A ``PipelineInput`` for each microbatch.

        Raises:
            TypeError: If the module does not support pipelining.
        """
        prev_stage_idx = None if self._info.is_current_stage_first else self._info.current_stage - 1
        next_stage_idx = None if self._info.is_current_stage_last else self._info.current_stage + 1

        module = self._module
        if not isinstance(module, ModuleSupportsPipelining):
            raise TypeError(
                f"The stage module ({type(module).__name__}) must implement the ModuleSupportsPipelining protocol."
            )

        self._forward_receiver = self._make_receiver(
            pipeline_inputs_per_microbatch,
            module,
            from_stage=prev_stage_idx,
            boundary=StageBoundary.incoming,
            requires_grad=has_backward,
        )
        self._forward_sender = self._make_sender(next_stage_idx)

        if has_backward:
            self._backward_comp = BackwardComputeHandler(
                stage_index=self._info.current_stage,
                module=module,
                has_input_peer=not self._info.is_current_stage_first,
            )
            self._backward_receiver = self._make_receiver(
                pipeline_inputs_per_microbatch,
                module,
                from_stage=next_stage_idx,
                boundary=StageBoundary.outgoing,
                requires_grad=False,
            )
            self._backward_sender = self._make_sender(prev_stage_idx)

        self._configured = True

    def _require_configured(self):
        if not self._configured:
            raise ValueError("The stage buffers are not configured. Call configure_buffers() first.")

    def _require_backward(self) -> BackwardComputeHandler[TPipelineInput, TStageTransfer]:
        if self._backward_comp is None:
            raise ValueError(
                "The stage is not configured for a backward pass. Call configure_buffers() with has_backward=True."
            )
        return self._backward_comp

    def set_local_fwd_input(self, inputs: TStageTransfer, microbatch_index: int):
        """Sets the forward inputs from the previous stage on the same rank.

        V-shape schedules use it, because the previous stage can live on the same rank.

        Args:
            inputs: The ``StageTransfer`` produced by the previous stage.
            microbatch_index: The microbatch identifier.

        Raises:
            ValueError: If the stage is not configured, or has no forward receiver (first stage).
        """
        self._require_configured()
        if self._forward_receiver is None:
            raise ValueError("The first stage has no forward receiver.")

        self._forward_receiver.set_inputs_local(inputs, microbatch_index)

    def get_produced_transfer(self, microbatch_index: int) -> TStageTransfer:
        """Returns the ``StageTransfer`` this (non-last) stage produced, to hand to the next stage.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The produced ``StageTransfer``.

        Raises:
            ValueError: If called on the last stage, which produces a ``PipelineOutput`` instead.
        """
        if self._info.is_current_stage_last:
            raise ValueError("The last stage produces a PipelineOutput, not a StageTransfer.")

        return cast(TStageTransfer, self._forward_comp.get_outputs(microbatch_index))

    def get_pipeline_output(self, microbatch_index: int) -> TPipelineOutput:
        """Returns the ``PipelineOutput`` this (last) stage produced, to hand to the callback.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The produced ``PipelineOutput``.

        Raises:
            ValueError: If called on a non-last stage, which produces a ``StageTransfer`` instead.
        """
        if not self._info.is_current_stage_last:
            raise ValueError("Only the last stage produces a PipelineOutput.")

        return cast(TPipelineOutput, self._forward_comp.get_outputs(microbatch_index))

    def pop_local_bwd_output(self, microbatch_index: int) -> TStageTransfer:
        """Retrieves and releases the input gradients to hand to the previous stage on the same rank.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The input gradients, shaped like the incoming ``StageTransfer``.

        Raises:
            ValueError: If the stage is not configured for backward passes.
        """
        self._require_configured()
        backward_comp = self._require_backward()

        return backward_comp.pop_for_sending(microbatch_index)

    def set_local_bwd_input(self, inputs: TStageTransfer, microbatch_index: int):
        """Sets the output gradients from the next stage on the same rank.

        Args:
            inputs: The gradients of the ``StageTransfer`` this stage produced.
            microbatch_index: The microbatch identifier.

        Raises:
            ValueError: If the stage is not configured for backward passes, or has no backward
                receiver (last stage).
        """
        self._require_configured()
        self._require_backward()
        if self._backward_receiver is None:
            raise ValueError("The last stage has no backward receiver: its backward pass starts from the loss.")

        self._backward_receiver.set_inputs_local(inputs, microbatch_index)

    def get_fwd_recv_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """Returns P2P ops to receive forward inputs for the given microbatch.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The list of P2P operations (empty if this stage has no forward receiver).

        Raises:
            ValueError: If the stage is not configured.
        """
        self._require_configured()
        if self._forward_receiver is None:
            return []

        return self._forward_receiver.receive(microbatch_index)

    def get_fwd_send_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """Returns P2P ops to send forward outputs for the given microbatch.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The list of P2P operations (empty if this stage has no forward sender).

        Raises:
            ValueError: If the stage is not configured.
        """
        self._require_configured()
        if self._forward_sender is None:
            return []

        return self._forward_sender.send(self.get_produced_transfer(microbatch_index))

    def get_bwd_recv_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """Returns P2P ops to receive output gradients for the given microbatch.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The list of P2P operations (empty if this stage has no backward receiver).
        """
        if self._backward_comp is None:
            return []

        self._require_configured()
        if self._backward_receiver is None:
            return []

        return self._backward_receiver.receive(microbatch_index)

    def get_bwd_send_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """Returns P2P ops to send input gradients for the given microbatch.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            The list of P2P operations (empty if this stage has no backward sender).
        """
        if self._backward_comp is None:
            return []

        self._require_configured()
        if self._backward_sender is None:
            return []

        bwd_result = self._backward_comp.pop_for_sending(microbatch_index)
        return self._backward_sender.send(bwd_result)

    def forward_one_chunk(
        self,
        microbatch_index: int,
        pipeline_inputs: TPipelineInput,
        pipeline_shared: TSharedInput,
    ):
        """Runs the forward pass for one microbatch.

        The first stage reads ``pipeline_inputs``. Other stages read the received transfer. The
        result is cached for the backward pass.

        Args:
            microbatch_index: The microbatch index.
            pipeline_inputs: The ``PipelineInput`` of the microbatch. Only the first stage uses it.
            pipeline_shared: The ``SharedInput`` passed to every stage.

        Raises:
            ValueError: If the stage is not configured.
        """
        self._require_configured()

        if self._forward_receiver is None:
            inputs = pipeline_inputs
        else:
            inputs = self._forward_receiver.pop_inputs(microbatch_index)

        self._forward_comp.run(microbatch_index=microbatch_index, inputs=inputs, shared=pipeline_shared)

    def backward_one_chunk(self, microbatch_index: int, loss: torch.Tensor | None = None, full_backward: bool = True):
        """Runs the backward pass for one microbatch.

        Args:
            microbatch_index: The microbatch index.
            loss: The loss tensor. Only the last stage uses it, and there it must be set.
            full_backward: If ``True``, computes gradients for the inputs and the weights. If
                ``False``, computes only the input gradients; ``backward_weight_one_chunk`` computes
                the weight gradients later.

        Raises:
            ValueError: If the stage is not configured for backward passes, or if ``loss`` is
                ``None`` on the last stage.
        """
        self._require_configured()
        backward_comp = self._require_backward()

        inputs, fwd_outputs = self._forward_comp.pop_inputs_outputs(microbatch_index)

        seed: BackwardSeed[TStageTransfer]
        if self._backward_receiver is None:
            # Last stage: the output does not leave the pipeline, so the backward pass starts from the loss.
            if loss is None:
                raise ValueError("The last stage needs a loss for the backward pass, but loss is None.")
            seed = BackwardSeedLoss(loss=loss)
        else:
            seed = BackwardSeedTransfer(
                outputs=cast(TStageTransfer, fwd_outputs),
                output_grads=self._backward_receiver.pop_inputs(microbatch_index),
            )

        if full_backward:
            backward_comp.backward_full(microbatch_index=microbatch_index, inputs=inputs, seed=seed)
        else:
            backward_comp.backward_input(microbatch_index=microbatch_index, inputs=inputs, seed=seed)

        # The last stage no longer needs its outputs for the backward pass. Detaching them frees the
        # autograd graph early. Views cannot be detached in place. Mirrors torch/distributed/pipelining/stage.py.
        if self._info.is_current_stage_last and not self._info.is_current_stage_first:
            for t in pytree.tree_leaves(fwd_outputs):
                if not t._is_view():  # noqa: SLF001 - PyTorch has no public check for views
                    t.detach_()

    def backward_weight_one_chunk(self, microbatch_index: int):
        """Runs the deferred weight backward for one microbatch.

        ``backward_one_chunk(..., full_backward=False)`` must run first for this microbatch.

        Args:
            microbatch_index: The microbatch index.

        Raises:
            ValueError: If the stage is not configured for backward passes.
        """
        self._require_configured()
        backward_comp = self._require_backward()

        backward_comp.backward_weight(microbatch_index=microbatch_index)

    def reset(self):
        """Releases the live receive buffers of this stage."""
        if self._forward_receiver is not None:
            self._forward_receiver.reset()
        if self._backward_receiver is not None:
            self._backward_receiver.reset()
