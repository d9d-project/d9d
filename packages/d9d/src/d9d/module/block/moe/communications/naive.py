import torch
from torch import Size

from d9d.kernel.moe import (
    fused_indices_to_multihot,
    moe_permute_with_probs,
    moe_unpermute_mask,
)
from d9d.module.block.moe.communications import ExpertCommunicationHandler
from d9d.module.block.moe.grouped_mm import ExpertTokenGroups


class NoCommunicationHandler(ExpertCommunicationHandler):
    """MoE communication handler for the case when all experts are on the local device.

    This handler does no communication. It only sorts the tokens by expert index.
    """

    def __init__(self, num_experts: int):
        """Constructs the ``NoCommunicationHandler`` object.

        Args:
            num_experts: Number of local experts.
        """
        self._num_experts = num_experts

        self._hidden_shape_before_permute: Size | None = None
        self._unpermute_mapping: torch.Tensor | None = None

    def dispatch(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, ExpertTokenGroups]:
        routing_map, routing_probs = fused_indices_to_multihot(topk_ids, topk_weights, self._num_experts)

        self._hidden_shape_before_permute = hidden_states.shape

        # Every token goes to exactly top-k experts, so the permuted row count is known without the counts.
        hidden_states, routing_probs, reverse_permute_map = moe_permute_with_probs(
            hidden_states, routing_probs, routing_map, num_out_tokens=topk_ids.numel()
        )

        self._unpermute_mapping = reverse_permute_map

        # The tokens are permuted by the routing map, so it gives the counts of the experts on the device.
        return hidden_states, routing_probs, ExpertTokenGroups(routing_map.sum(dim=0))

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._unpermute_mapping is None:
            raise ValueError("Cannot run combine() before dispatch(). Call dispatch() first.")

        hidden_states = moe_unpermute_mask(
            hidden_states,
            self._unpermute_mapping,
            restore_shape=self._hidden_shape_before_permute,
        )

        self._unpermute_mapping = None
        self._hidden_shape_before_permute = None

        return hidden_states
