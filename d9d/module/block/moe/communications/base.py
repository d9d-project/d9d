import abc

import torch


class ExpertCommunicationHandler(abc.ABC):
    """Abstract base class for Mixture-of-Experts communication strategies."""

    @abc.abstractmethod
    def dispatch(
            self,
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares and routes local hidden states to their target experts (possibly on other workers).

        This process involves:

        1. All-to-All Communication: Transfers hidden states to workers containing the assigned experts. States
        assigned to multiple experts are replicated.

        2. Permutation: Sorts tokens by expert ID to prepare for Grouped GEMM.

        Args:
            hidden_states: Input tokens. Shape: `(num_tokens, hidden_size)`.
            topk_ids: Indices of the top-k experts selected for each token. Shape: `(num_tokens, k)`.
            topk_weights: Routing weights associated with the selected experts. Shape: `(num_tokens, k)`.

        Returns:
            A tuple containing:

            - Permuted hidden states received by this rank. Shape: `(num_received_tokens, hidden_size)`.
            - Permuted weights matching the hidden states order. Shape: `(num_received_tokens)`.
            - Expert count tensor indicating how many tokens each local expert received. Shape: `(num_local_experts)`.
        """

        ...

    @abc.abstractmethod
    def combine(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Restores hidden states to their original order and location.

        Undoes the permutation and performs the reverse All-to-All communication
        to return processed results to the workers that originated the requests.

        Args:
            hidden_states: The processed hidden states. Shape: `(num_received_tokens, hidden_size)`.

        Returns:
            The combined hidden states with the original shape and order. Shape: `(num_tokens, hidden_size)`.
        """
        ...
