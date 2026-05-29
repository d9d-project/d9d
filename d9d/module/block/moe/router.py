import dataclasses

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

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


class SigmoidGroupedTopKRouter(nn.Module, ModuleLateInit):
    """
    Selects top-K experts using sigmoid-activated scores with hierarchical group selection.

    Used by the DeepSeek-V3 family (DSv3, GLM-4.6, Kimi K2/K2.6, MiniMax-M2, Mistral Large/Small 4).

    Selection procedure per token:
      1. Gate logits → sigmoid → raw per-expert probabilities (float32).
      2. Add ``e_score_correction_bias`` to get selection scores (bias is not reflected in
         the final expert weights — it is selection-only for load-balancing purposes).
      3. Divide experts into ``n_group`` equal groups; compute each group's score as the sum
         of its top-2 expert selection scores.
      4. Keep the ``topk_group`` highest-scoring groups; mask out all other experts.
      5. From the surviving experts run a final top-``top_k`` selection.
      6. Gather weights from the unbiased sigmoid scores (step 1), not the biased scores.
      7. Optionally renormalise to sum=1, then multiply by ``routed_scaling_factor``.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool,
    ) -> None:
        """
        Constructs the SigmoidGroupedTopKRouter.

        Args:
            dim: Input feature dimensionality.
            num_experts: Total number of routed experts. Must be divisible by ``n_group``.
            top_k: Number of experts to select per token.
            n_group: Number of expert groups. Experts are partitioned into contiguous groups
                of size ``num_experts // n_group``.
            topk_group: Number of groups whose experts are candidates for final top-k.
            routed_scaling_factor: Scalar multiplied onto the final expert weights.
            norm_topk_prob: If True, renormalise selected weights to sum to 1 before scaling.
        """
        super().__init__()
        if num_experts % n_group != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by n_group ({n_group})")

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.e_score_correction_bias = nn.Buffer(
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=True,
        )

        self._num_experts = num_experts
        self._top_k = top_k
        self._n_group = n_group
        self._topk_group = topk_group
        self._experts_per_group = num_experts // n_group
        self._routed_scaling_factor = routed_scaling_factor
        self._norm_topk_prob = norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> RoutingResult:
        """
        Calculates routing decisions for the input tokens.

        Args:
            hidden_states: Input tokens. Shape: ``(num_tokens, dim)``.

        Returns:
            The routing result with selected expert indices and their (possibly scaled) weights.
        """
        num_tokens = hidden_states.shape[0]

        scores = F.linear(hidden_states.float(), self.gate.weight.float()).sigmoid()

        bias = self.e_score_correction_bias
        if isinstance(bias, DTensor):
            bias = bias.to_local()
        scores_for_sel = scores + bias

        # Group scoring: sum of top-2 expert scores within each group
        grouped = scores_for_sel.view(num_tokens, self._n_group, self._experts_per_group)
        group_scores = grouped.topk(min(2, self._experts_per_group), dim=-1)[0].sum(dim=-1)

        # Select topk_group groups
        group_idx = torch.topk(group_scores, k=self._topk_group, dim=-1, sorted=False)[1]

        # Build expert-level boolean mask from selected groups
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        expert_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, -1, self._experts_per_group)
            .reshape(num_tokens, self._num_experts)
            .bool()
        )

        # Final top-k within surviving experts
        masked_scores = scores_for_sel.masked_fill(~expert_mask, float("-inf"))
        selected_indices = torch.topk(masked_scores, k=self._top_k, dim=-1, sorted=False)[1]

        # Weights from unbiased scores (bias must not affect the FFN computation)
        selected_weights = scores.gather(dim=-1, index=selected_indices)

        if self._norm_topk_prob:
            selected_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + 1e-20)

        selected_weights = selected_weights * self._routed_scaling_factor

        return RoutingResult(selected_expert_indices=selected_indices, selected_probabilities=selected_weights)

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.gate.reset_parameters()
        nn.init.zeros_(self.e_score_correction_bias)
