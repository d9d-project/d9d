import torch
import torch.nn.functional as F
from torch import nn

from d9d.module.base import ModuleLateInit


class TopKRouter(nn.Module, ModuleLateInit):
    """
    Selects the top-K experts based on a learned gating mechanism.

    This router:

    1. Projects input tokens into expert space
    2. Applies softmax, optionally adds expert bias to influence selection
    3. Selects the experts with the highest probabilities
    4. Selected probabilities are then re-normalized to sum to 1 if needed.
    """

    def __init__(
            self,
            dim: int,
            num_experts: int,
            top_k: int,
            renormalize_probabilities: bool,
            enable_expert_bias: bool = False
    ):
        """
        Constructs the TopKRouter.

        Args:
            dim: Input feature dimensionality.
            num_experts: Total number of experts to choose from.
            top_k: Number of experts to select for each token.
            renormalize_probabilities: If True, probabilities of selected experts will be renormalized to sum up to 1
            enable_expert_bias: If True, adds a bias term to the routing scores before top-k selection. This can be
                used for loss-free load balancing.
        """

        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)

        if enable_expert_bias:
            self.expert_bias = nn.Buffer(
                torch.empty(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

        self._num_experts = num_experts
        self._top_k = top_k
        self._renormalize_probabilities = renormalize_probabilities

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates routing decisions for the input tokens.

        Args:
            hidden_states: Input tokens. Shape: `(num_tokens, dim)`.

        Returns:
            A tuple containing:

            - Selected expert indices. Shape: `(num_tokens, top_k)`.
            - Normalized routing weights for the selected experts. Shape: `(num_tokens, top_k)`.
        """

        # scores shape (bs*slen, num_experts)

        # gate
        scores = self.gate(hidden_states)

        # and now do softmax (before top-k to be able to apply expert bias)
        scores = F.softmax(scores, dim=-1, dtype=torch.float32)

        # select top-k
        if self.expert_bias is None:
            scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=-1
            )
        else:
            _, selected_experts_indices = torch.topk(
                scores + self.expert_bias, k=self.top_k, dim=-1
            )
            scores = scores.gather(dim=-1, index=selected_experts_indices)

        # re-normalize scores
        denominator = scores.sum(dim=-1, keepdim=True) + 1e-20
        scores = scores / denominator

        return selected_experts_indices, scores

    def reset_parameters(self):
        """Resets module parameters."""
        if self.expert_bias is not None:
            nn.init.zeros_(self.expert_bias)

        self.gate.reset_parameters()
