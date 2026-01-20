import dataclasses

import torch
import torch.distributed as dist


@dataclasses.dataclass(kw_only=True, slots=True)
class ReceiveStageInput:
    """
    Instruction to receive a specific tensor from a previous stage (or next stage during backward).

    Attributes:
        name: A unique identifier for the communication operation.
        from_stage: The stage index sending the data.
        buffer: The pre-allocated tensor buffer where data will be received.
    """

    name: str
    from_stage: int
    buffer: torch.Tensor


@dataclasses.dataclass
class StartStageInput:
    """
    Instruction indicating that the input for this stage does not come from communication
    (e.g., this is the first stage receiving data loader inputs).
    """


StageInput = ReceiveStageInput | StartStageInput


@dataclasses.dataclass(kw_only=True, slots=True)
class SendStageOutput:
    """
    Instruction to send a specific tensor to a next stage (or previous if backward).

    Attributes:
        to_stage: The stage index receiving the data.
    """

    to_stage: int


@dataclasses.dataclass
class EndStageOutput:
    """
    Instruction indicating that the output of this stage is not sent anywhere
    (e.g., this is the last stage computing loss).
    """


StageOutput = SendStageOutput | EndStageOutput


class StageCommunicationHandler:
    """
    Manages Point-to-Point (P2P) communication descriptors for a specific data flow direction within a pipeline stage.

    This class handles the creation of P2P operations (send/recv) across multiple microbatches,
    managing buffers and mapping logical stage indices to physical ranks.
    """

    def __init__(
            self,

            name: str,
            stage_index: int,
            num_microbatches: int,

            input_stage_index: int | None,
            input_args: dict[str, torch.Tensor],

            output_stage_index: int | None,
            output_args: dict[str, torch.Tensor],

            stage_idx_to_host_rank: dict[int, int],
            group: dist.ProcessGroup
    ):
        """
        Constructs a StageCommunicationHandler object.

        Args:
            name: Name prefix for this handler (e.g., 'fwd', 'bwd').
            stage_index: The logical index of the current stage.
            num_microbatches: Total number of microbatches ("chunks") to schedule.
            input_stage_index: The logical index of the stage providing inputs, or None if inputs are local.
            input_args: Metadata (shapes/dtypes) for input tensors.
            output_stage_index: The logical index of the stage consuming outputs, or None if outputs are terminal.
            output_args: Metadata (shapes/dtypes) for output tensors.
            stage_idx_to_host_rank: Mapping from logical stage indices to physical world ranks.
            group: The process group strictly for pipeline communication.
        """

        self._input_handlers = self._build_inputs(
            name=name,
            stage_index=stage_index,
            num_microbatches=num_microbatches,
            input_stage_index=input_stage_index,
            input_args=input_args
        )
        self._output_handlers = self._build_outputs(
            output_stage_index=output_stage_index,
            output_args=output_args
        )

        self._stage_idx_to_host_rank = stage_idx_to_host_rank
        self._group = group

    @staticmethod
    def _build_inputs(
            name: str,
            stage_index: int,
            num_microbatches: int,
            input_stage_index: int | None,
            input_args: dict[str, torch.Tensor]
    ) -> dict[int, dict[str, StageInput]]:
        handlers = {}

        for chunk_id in range(num_microbatches):
            handlers[chunk_id] = {}
            for input_name, input_tensor_meta in input_args.items():
                if input_stage_index is None:
                    handlers[chunk_id][input_name] = StartStageInput()
                else:
                    handlers[chunk_id][input_name] = ReceiveStageInput(
                        name=f"{name}_recv_from_{input_stage_index}_to_{stage_index}[{chunk_id}][{input_name}]",
                        from_stage=input_stage_index,
                        buffer=torch.empty(
                            input_tensor_meta.size(),
                            dtype=input_tensor_meta.dtype,
                            layout=input_tensor_meta.layout,
                            device="cuda"  # force device
                        )
                    )
        return handlers

    @staticmethod
    def _build_outputs(
            output_stage_index: int | None,
            output_args: dict[str, torch.Tensor]
    ) -> dict[str, StageOutput]:
        handlers = {}

        for output_name in output_args:
            if output_stage_index is None:
                handlers[output_name] = EndStageOutput()
            else:
                handlers[output_name] = SendStageOutput(
                    to_stage=output_stage_index
                )
        return handlers

    def set_input_requires_grad_(self, requires_grad: bool):
        """
        Sets the `requires_grad` flag for all internal input buffers.

        Typically used to enable gradient flow from backward stages to forward stages.

        Args:
            requires_grad: Whether the buffers should require gradients.
        """

        for inputs in self._input_handlers.values():
            for info in inputs.values():
                if isinstance(info, ReceiveStageInput):
                    info.buffer.requires_grad_(requires_grad)

    def set_inputs_local(self, inputs: dict[str, torch.Tensor], microbatch_index: int):
        """
        Manually fills the input buffer for a specific microbatch with local data.

        This is used when the stage is the first in the pipeline or receives data
        from a dataloader rather than via network communication.

        Args:
            inputs: Dictionary of input tensors.
            microbatch_index: The microbatch identifier.
        """

        for input_name, input_value in inputs.items():
            prev_requires_grad = self._input_handlers[microbatch_index][input_name].buffer.requires_grad
            self._input_handlers[microbatch_index][input_name].buffer = input_value.detach().requires_grad_(
                prev_requires_grad)

    def get_inputs(self, microbatch_index: int) -> dict[str, torch.Tensor]:
        """
        Retrieves the input tensors for a specific microbatch from the internal buffers.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            Dictionary mapping input names to tensors.
        """

        return {
            input_name: input_info.buffer
            for input_name, input_info
            in self._input_handlers[microbatch_index].items()
        }

    def create_receive_ops(self, microbatch_index: int) -> list[dist.P2POp]:
        """
        Generates the PyTorch P2P receive operations for a specific microbatch.

        Args:
            microbatch_index: The microbatch identifier.

        Returns:
            A list of `dist.P2POp` objects configured for `dist.irecv`.
        """

        ops = []

        inputs = self._input_handlers[microbatch_index]
        # sort ops by parameter names to ensure receive ops are ordered the same for send and recv
        for _input_name, input_info in sorted(inputs.items(), key=lambda x: x[0]):
            match input_info:
                case StartStageInput():
                    pass
                case ReceiveStageInput():
                    peer_rank = self._stage_idx_to_host_rank[input_info.from_stage]
                    peer_global_rank = dist.get_global_rank(self._group, peer_rank)
                    op = dist.P2POp(dist.irecv, input_info.buffer, peer_global_rank, self._group)
                    ops.append(op)
                case _:
                    raise ValueError()

        return ops

    def create_send_ops(self, send_contents: dict[str, torch.Tensor]) -> list[dist.P2POp]:
        """
        Generates the PyTorch P2P send operations for the provided tensors.

        Args:
            send_contents: Dictionary of tensors to send.

        Returns:
            A list of `dist.P2POp` objects configured for `dist.isend`.
        """

        ops = []

        # sort ops by parameter names to ensure receive ops are ordered the same for send and recv
        for output_name, output_tensor in sorted(send_contents.items(), key=lambda x: x[0]):
            output_info = self._output_handlers[output_name]

            match output_info:
                case EndStageOutput():
                    pass
                case SendStageOutput():
                    peer_rank = self._stage_idx_to_host_rank[output_info.to_stage]
                    peer_global_rank = dist.get_global_rank(self._group, peer_rank)
                    op = dist.P2POp(dist.isend, output_tensor, peer_global_rank, self._group)
                    ops.append(op)
                case _:
                    raise ValueError()

        return ops

    def reset(self):
        """Resets the internal state, specifically clearing gradients on input buffers."""

        for inp_handlers in self._input_handlers.values():
            for inp_handler in inp_handlers.values():
                if isinstance(inp_handler, ReceiveStageInput):
                    inp_handler.buffer.grad = None
