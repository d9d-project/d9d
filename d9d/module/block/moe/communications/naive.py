from typing import cast

import torch
from torch import Size

from d9d.kernel.moe import (
    fused_indices_to_multihot,
    moe_permute_with_probs,
    moe_unpermute_mask,
)
from d9d.module.block.moe.communications import ExpertCommunicationHandler


class NoCommunicationHandler(ExpertCommunicationHandler):
    """
    Handles MoE routing within a single device or when no cross-device routing is needed.

    This handler does not perform network operations. It only permutes elements
    mostly for local logical grouping or debugging.
    """

    def __init__(self, num_experts: int):
        """Constructs the NoCommunicationHandler."""
        self._num_experts = num_experts

        self._hidden_shape_before_permute: Size | None = None
        self._unpermute_mapping: torch.Tensor | None = None

    def dispatch(
            self,
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens_per_expert = torch.bincount(topk_ids.flatten(), minlength=self._num_experts).cpu()

        routing_map, routing_probs = fused_indices_to_multihot(
            topk_ids, topk_weights, self._num_experts
        )

        self._hidden_shape_before_permute = hidden_states.shape

        hidden_states, routing_probs, reverse_permute_map = moe_permute_with_probs(
            hidden_states,
            routing_probs,
            routing_map,
            num_out_tokens=cast(int, tokens_per_expert.sum().item())
        )

        self._unpermute_mapping = reverse_permute_map

        return hidden_states, routing_probs, tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._unpermute_mapping is None:
            raise ValueError("Cannot run combine before running dispatch!")

        hidden_states = moe_unpermute_mask(
            hidden_states,
            self._unpermute_mapping,
            restore_shape=self._hidden_shape_before_permute,
        )

        self._unpermute_mapping = None
        self._hidden_shape_before_permute = None

        return hidden_states
