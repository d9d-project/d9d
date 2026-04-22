import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from d9d.module.base import ModuleLateInit


@dataclasses.dataclass(kw_only=True, slots=True)
class RoutingResult:
    """
    Represents the result of a routing operation to select experts.

    Attributes:
        selected_expert_indices: Indices of the chosen experts per token.
        selected_probabilities: Probabilities associated with the chosen experts.
    """

    selected_expert_indices: torch.Tensor
    selected_probabilities: torch.Tensor


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
        self, dim: int, num_experts: int, top_k: int, renormalize_probabilities: bool, enable_expert_bias: bool = False
    ) -> None:
        """
        Constructs the TopKRouter.

        Args:
            dim: Input feature dimensionality.
            num_experts: Total number of experts to choose from.
            top_k: Number of experts to select for each token.
            renormalize_probabilities: If True, probabilities of selected experts will be renormalized to sum up to 1.
            enable_expert_bias: If True, adds a bias term to the routing scores before top-k selection. This can be
                used for loss-free load balancing.
        """

        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)

        self.expert_bias: nn.Buffer | None
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

    def forward(self, hidden_states: torch.Tensor) -> RoutingResult:
        """
        Calculates routing decisions for the input tokens.

        Args:
            hidden_states: Input tokens. Shape: `(num_tokens, dim)`.

        Returns:
            The routing result containing the indices of the selected experts and their corresponding probabilities.
        """

        # scores shape (bs*slen, num_experts)

        # gate
        scores = self.gate(hidden_states)

        # and now do softmax (before top-k to be able to apply expert bias)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)

        # select top-k
        if self.expert_bias is None:
            selected_probs, selected_experts_indices = torch.topk(probs, k=self._top_k, dim=-1)
        else:
            _, selected_experts_indices = torch.topk(probs + self.expert_bias, k=self._top_k, dim=-1)
            selected_probs = probs.gather(dim=-1, index=selected_experts_indices)

        # re-normalize scores
        if self._renormalize_probabilities:
            denominator = selected_probs.sum(dim=-1, keepdim=True) + 1e-20
            selected_probs = selected_probs / denominator

        return RoutingResult(selected_expert_indices=selected_experts_indices, selected_probabilities=selected_probs)

    def reset_parameters(self) -> None:
        """
        Resets module parameters.
        """
        if self.expert_bias is not None:
            nn.init.zeros_(self.expert_bias)

        self.gate.reset_parameters()
