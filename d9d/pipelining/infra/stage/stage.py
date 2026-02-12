from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from d9d.pipelining.api import ModuleSupportsPipelining, PipelineStageInfo

from .communications import StageCommunicationHandler
from .computations import BackwardComputeHandler, ForwardComputeHandler


class PipelineStage:
    """
    Represents a single structural stage in a Pipelined Model.

    This class acts as an orchestrator that combines `StageCommunicationHandler` (for I/O)
    and `Forward/BackwardComputeHandler` (for execution). It abstracts away the complexity
    of buffer management, distributed communication, and gradient calculation from the scheduler.
    """

    def __init__(
        self,
        info: PipelineStageInfo,
        module: nn.Module,
        group: dist.ProcessGroup,
        stage_to_host_topology: dict[int, int],
    ):
        """
        Constructs a PipelineStage object.

        Args:
            info: Metadata about the stage (index, total stages).
            module: The PyTorch module executed by this stage.
            group: The distributed process group for pipeline communications.
            stage_to_host_topology: Dict mapping stage ID to PP rank hosting it.
        """

        self._info = info
        self._module = module
        self._group = group
        self._stage_to_host_topology = stage_to_host_topology

        self._has_backward = False

        self._forward_comm: StageCommunicationHandler | None = None
        self._backward_comm: StageCommunicationHandler | None = None

        self._forward_comp = ForwardComputeHandler(stage_index=info.current_stage, module=module)
        self._backward_comp = BackwardComputeHandler(stage_index=info.current_stage, module=module)

    @property
    def info(self) -> PipelineStageInfo:
        return self._info

    def configure_buffers(self, num_microbatches: int, has_backward: bool, pipeline_inputs: dict[str, torch.Tensor]):
        """
        Initializes the communication handlers and buffers for the stage.

        This must be called before execution to establish P2P buffer sizes and directions.

        Args:
            num_microbatches: Total number of microbatches to process.
            has_backward: Does this pipeline stage should store info for a backward pass
            pipeline_inputs: Pipeline input data.
        """

        self._has_backward = has_backward

        prev_stage_idx = None if self._info.is_current_stage_first else self._info.current_stage - 1
        next_stage_idx = None if self._info.is_current_stage_last else self._info.current_stage + 1

        with torch.device("meta"):
            if not isinstance(self._module, ModuleSupportsPipelining):
                raise TypeError("Module does not implement ModuleSupportsPipelining protocol")
            inputs_meta = self._module.infer_stage_inputs_from_pipeline_inputs(
                inputs=pipeline_inputs, n_microbatches=num_microbatches
            )
            outputs_meta = self._module.infer_stage_outputs_from_pipeline_inputs(
                inputs=pipeline_inputs, n_microbatches=num_microbatches
            )

        self._forward_comm = StageCommunicationHandler(
            name="fwd",
            stage_index=self._info.current_stage,
            num_microbatches=num_microbatches,
            input_stage_index=prev_stage_idx,
            input_args=inputs_meta,
            output_stage_index=next_stage_idx,
            output_args=outputs_meta,
            group=self._group,
            stage_idx_to_host_rank=self._stage_to_host_topology,
        )
        self._forward_comm.set_input_requires_grad_(requires_grad=has_backward)

        if has_backward:
            # for grad - current stage receives OUTPUTS as inputs and sends INPUTS as outputs
            # because it is reversed forward
            self._backward_comm = StageCommunicationHandler(
                name="bwd",
                stage_index=self._info.current_stage,
                num_microbatches=num_microbatches,
                input_stage_index=next_stage_idx,
                input_args=outputs_meta,
                output_stage_index=prev_stage_idx,
                output_args=inputs_meta,
                group=self._group,
                stage_idx_to_host_rank=self._stage_to_host_topology,
            )
        else:
            self._backward_comm = None

    def set_local_fwd_input(self, inputs: dict[str, torch.Tensor], microbatch_index: int):
        """
        Sets local forward inputs manually.

        Used for the V-shape schedulers.
        """

        if self._forward_comm is None:
            raise ValueError("You must configure stage buffers first")

        self._forward_comm.set_inputs_local(inputs, microbatch_index)

    def get_local_fwd_output(self, microbatch_index: int) -> dict[str, torch.Tensor]:
        return self._forward_comp.get_outputs(microbatch_index)

    def pop_local_bwd_output(self, microbatch_index: int) -> dict[str, torch.Tensor]:
        """
        Retrieves local backward outputs (gradients).
        """

        if not self._has_backward:
            raise ValueError()

        return self._backward_comp.pop_for_sending(microbatch_index)

    def set_local_bwd_input(self, inputs: dict[str, torch.Tensor], microbatch_index: int):
        """
        Sets local backward inputs (output gradients) manually.
        """

        if not self._has_backward:
            raise ValueError()

        if self._backward_comm is None:
            raise ValueError("You must configure stage buffers first")

        self._backward_comm.set_inputs_local(inputs, microbatch_index)

    def get_fwd_recv_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """Returns P2P ops to receive forward inputs for the given microbatch."""

        if self._forward_comm is None:
            raise ValueError("You must configure stage buffers first")

        return self._forward_comm.create_receive_ops(microbatch_index)

    def get_fwd_send_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """Returns P2P ops to send forward outputs for the given microbatch."""

        if self._forward_comm is None:
            raise ValueError("You must configure stage buffers first")

        fwd_result = self._forward_comp.get_outputs(microbatch_index)
        return self._forward_comm.create_send_ops(fwd_result)

    def get_bwd_recv_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """Returns P2P ops to receive backward gradients for the given microbatch."""

        if not self._has_backward:
            return []

        if self._backward_comm is None:
            raise ValueError("You must configure stage buffers first")

        return self._backward_comm.create_receive_ops(microbatch_index)

    def get_bwd_send_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """Returns P2P ops to send backward gradients for the given microbatch."""

        if not self._has_backward:
            return []

        if self._backward_comm is None:
            raise ValueError("You must configure stage buffers first")

        bwd_result = self._backward_comp.pop_for_sending(microbatch_index)
        return self._backward_comm.create_send_ops(bwd_result)

    def forward_one_chunk(
        self,
        microbatch_index: int,
        pipeline_inputs: dict[str, torch.Tensor],
        pipeline_kwargs: dict[str, Any] | None = None,
    ):
        """
        Executes a forward pass for a single microbatch chunk.

        Fetches inputs from the communication buffer (or `pipeline_inputs` if first stage),
        runs the computation, and caches the result.

        Args:
            microbatch_index: The microbatch index.
            pipeline_inputs: Inputs provided locally (only used if this is the first stage).
            pipeline_kwargs: Additional arguments for the module.

        Returns:
            The output tensors of the forward pass.
        """

        if self._forward_comm is None:
            raise ValueError("You must configure stage buffers first")

        if self._info.is_current_stage_first:
            inputs = pipeline_inputs
        else:
            inputs = self._forward_comm.get_inputs(microbatch_index)

        kwargs = pipeline_kwargs or {}

        self._forward_comp.run(microbatch_index=microbatch_index, inputs=inputs, kwargs=kwargs)

    def backward_one_chunk(self, microbatch_index: int, loss: torch.Tensor | None = None, full_backward: bool = True):
        """
        Executes a backward pass for a single microbatch chunk.

        Can perform either a full backward or just the input gradients (if `full_backward=False`).
        It fetches required data from forward cache and communication buffers.

        Args:
            microbatch_index: The microbatch index.
            loss: The loss tensor (only used if this is the last stage).
            full_backward: If True, computes grads for inputs and weights. If False, only for inputs.
        """

        if not self._has_backward:
            raise ValueError()

        if self._backward_comm is None:
            raise ValueError("You must configure stage buffers first")

        inputs, fwd_outputs = self._forward_comp.pop_inputs_outputs(microbatch_index)

        outputs: dict[str, torch.Tensor]
        outputs_grad: dict[str, torch.Tensor] | None

        if self._info.is_current_stage_last:
            if loss is None:
                raise ValueError("Cannot perform backward on last stage without loss specified")
            outputs = {"loss": loss}
            outputs_grad = None
        else:
            outputs = fwd_outputs
            outputs_grad = self._backward_comm.get_inputs(microbatch_index)

        if full_backward:
            self._backward_comp.backward_full(
                microbatch_index=microbatch_index, inputs=inputs, outputs=outputs, outputs_grad=outputs_grad
            )
        else:
            self._backward_comp.backward_input(
                microbatch_index=microbatch_index, inputs=inputs, outputs=outputs, outputs_grad=outputs_grad
            )

        if self._info.is_current_stage_last and not self._info.is_current_stage_first:
            for t in fwd_outputs.values():
                if not t._is_view():  # noqa: SLF001
                    t.detach_()

    def backward_weight_one_chunk(self, microbatch_index: int):
        """
        Executes the weight gradient accumulation part of the backward pass.

        This assumes `backward_one_chunk(..., full_backward=False)` was already called
        for this microbatch.

        Args:
            microbatch_index: The microbatch index.
        """

        if not self._has_backward:
            raise ValueError()

        self._backward_comp.backward_weight(microbatch_index=microbatch_index)

    def reset(self):
        """Resets the internal state of communication handlers, clearing gradients on buffers."""

        if self._forward_comm is not None:
            self._forward_comm.reset()
        if self._backward_comm is not None:
            self._backward_comm.reset()
