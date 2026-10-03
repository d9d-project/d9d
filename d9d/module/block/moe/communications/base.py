import abc

import torch


class ExpertCommunicationHandler(abc.ABC):
    """Abstract base class for Mixture-of-Experts communication strategies."""

    @abc.abstractmethod
    def dispatch(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Routes local hidden states to their target experts, which can be on other ranks.

        Tokens assigned to several experts are replicated. The received tokens are sorted by expert index,
        as grouped GEMM requires.

        Args:
            hidden_states: Input tokens. Shape: ``(num_tokens, hidden_size)``.
            topk_ids: Indices of the experts selected for each token. Shape: ``(num_tokens, top_k)``.
            topk_weights: Routing weights of the selected experts. Shape: ``(num_tokens, top_k)``.

        Returns:
            A tuple of:

            - Sorted hidden states received by this rank. Shape: ``(num_received_tokens, hidden_size)``.
            - Routing weights in the same order. Shape: ``(num_received_tokens,)``.
            - CPU tensor with the number of tokens each local expert received. Shape: ``(num_local_experts,)``.
        """
        ...

    @abc.abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns processed hidden states to their original order and rank.

        Replicated tokens are summed. Must be called after ``dispatch``.

        Args:
            hidden_states: Processed hidden states. Shape: ``(num_received_tokens, hidden_size)``.

        Returns:
            The combined hidden states in the original order. Shape: ``(num_tokens, hidden_size)``.
        """
        ...
