import torch
from deep_ep import Buffer, EventOverlap

from d9d.kernel.moe.indices_to_multihot import fused_indices_to_multihot
from d9d.kernel.moe.permute_with_probs import moe_permute_with_probs, moe_unpermute_mask
from d9d.module.block.moe.communications import ExpertCommunicationHandler

# see https://github.com/deepseek-ai/DeepEP/blob/main/README.md for examples
# TODO: implement computation/communication overlap for PP case

_buffer: Buffer | None = None


def get_hidden_state_bytes(x: torch.Tensor) -> int:
    """
    Calculates the byte size of a hidden state tensor row.

    Args:
        x: Input tensor. Shape: `(?, hidden_size)`.
    """

    return x.size(1) * max(x.element_size(), 2)


def init_deepep_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """
    Initializes or expands the global DeepEP communication buffer.

    Checks if the existing buffer is sufficient for the required hidden dimension
    and process group size. If not, it allocates a new buffer.

    Args:
        group: The process group intended for communication.
        hidden_bytes: Size of a single hidden state vector in bytes.
    """

    global _buffer  # noqa: PLW0603
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
            Buffer.get_dispatch_config(group.size()),
            Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    if (
            _buffer is None
            or _buffer.group != group
            or _buffer.num_nvl_bytes < num_nvl_bytes
            or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)


class DeepEpDispatch(torch.autograd.Function):
    """Autograd function for the DeepEP Dispatch operation."""

    @staticmethod
    def forward(
            ctx,
            x: torch.Tensor,
            topk_idx: torch.Tensor,
            topk_weights: torch.Tensor,
            num_experts: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list,
        tuple,
        EventOverlap
    ]:
        previous_event = Buffer.capture()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event
        ) = _buffer.get_dispatch_layout(
            topk_idx, num_experts,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True
        )

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            event
        ) = _buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True
        )

        event.current_stream_wait()

        num_recv_tokens_per_expert_list = torch.tensor(num_recv_tokens_per_expert_list)

        ctx.handle = handle

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle
        )

    @staticmethod
    def backward(
            ctx,
            grad_recv_x: torch.Tensor,
            grad_recv_topk_idx: torch.Tensor,
            grad_recv_topk_weights: torch.Tensor,
            grad_num_recv_tokens_per_expert_list,
            grad_handle
    ):
        handle = ctx.handle

        prev_event = Buffer.capture()

        (
            combined_grad_x,
            combined_grad_recv_topk_weights,
            event
        ) = _buffer.combine(
            grad_recv_x.contiguous(),
            handle,
            topk_weights=grad_recv_topk_weights,
            async_finish=True,
            previous_event=prev_event,
            allocate_on_comm_stream=True
        )

        event.current_stream_wait()

        return combined_grad_x, None, combined_grad_recv_topk_weights, None


class DeepEpCombine(torch.autograd.Function):
    """Autograd function for the DeepEP Combine operation."""

    @staticmethod
    def forward(
            ctx,
            x: torch.Tensor,
            handle
    ):
        previous_event = Buffer.capture()

        combined_x, _, event = _buffer.combine(
            x,
            handle,
            async_finish=True,
            previous_event=previous_event,
            allocate_on_comm_stream=True
        )

        event.current_stream_wait()

        ctx.handle = handle

        return combined_x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        handle = ctx.handle

        previous_event = Buffer.capture()

        grad_x, _, _, _, _, event = _buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
            async_finish=True,
            previous_event=previous_event,
            allocate_on_comm_stream=True
        )

        event.current_stream_wait()

        return grad_x, None


class DeepEpCommunicationHandler(ExpertCommunicationHandler):
    """Handles MoE communication using the high-performance DeepEP library."""

    def __init__(self, num_experts: int):
        """Constructs the DeepEpCommunicationHandler."""

        self._num_experts = num_experts
        self._num_experts_per_shard = None  # late-initialization

        # == fields saved for post-dispatch ==

        self._handle = None
        self._hidden_shape_before_permute = None
        self._unpermute_mapping = None

    def setup(self, group: torch.distributed.ProcessGroup, hidden_size: int, hidden_dtype: torch.dtype):
        """
        Initializes the backend buffer and calculates expert sharding.

        Args:
            group: The process group containing all experts.
            hidden_size: Dimensionality of the hidden states.
            hidden_dtype: Data type of the hidden states.
        """

        init_deepep_buffer(group, hidden_size * hidden_dtype.itemsize)

        if self._num_experts % group.size() != 0:
            raise ValueError("num_experts must be divisible by distributed group size")

        self._num_experts_per_shard = self._num_experts // group.size()

    def dispatch(
            self,
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            hidden_states,
            topk_ids,
            topk_weights,
            tokens_per_expert,
            handle
        ) = DeepEpDispatch.apply(
            hidden_states,
            topk_ids,
            topk_weights,
            self._num_experts
        )

        routing_map, routing_probs = fused_indices_to_multihot(
            topk_ids, topk_weights, self._num_experts_per_shard
        )

        self._hidden_shape_before_permute = hidden_states.shape

        hidden_states, routing_probs, reverse_permute_map = moe_permute_with_probs(
            hidden_states,
            routing_probs,
            routing_map,
            num_out_tokens=tokens_per_expert.sum().item()
        )

        self._handle = handle
        self._unpermute_mapping = reverse_permute_map

        return hidden_states, routing_probs, tokens_per_expert

    def combine(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self._handle is None:
            raise ValueError("you fucked up moe communication order: you should dispatch first and after that combine")

        hidden_states = moe_unpermute_mask(
            hidden_states,
            self._unpermute_mapping,
            restore_shape=self._hidden_shape_before_permute,
        )

        hidden_states = DeepEpCombine.apply(
            hidden_states,
            self._handle
        )

        self._handle = None
        self._unpermute_mapping = None
        self._hidden_shape_before_permute = None

        return hidden_states
