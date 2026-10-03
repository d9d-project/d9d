import torch
from torch import nn

from d9d.kernel.swiglu import silu_mul
from d9d.module.base import ModuleLateInit

from .grouped_linear import GroupedLinear


class GroupedSwiGLU(nn.Module, ModuleLateInit):
    """Set of SwiGLU experts that run with grouped GEMM.

    Each expert computes ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``. All experts run in one grouped GEMM
    per projection, without padding or masking.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, num_experts: int):
        """Constructs the ``GroupedSwiGLU`` object.

        Args:
            hidden_dim: Hidden size of the input and output.
            intermediate_dim: Intermediate size of each expert.
            num_experts: Number of experts held by this module.
        """
        super().__init__()
        self._num_experts = num_experts

        self.gate_proj = GroupedLinear(num_experts, hidden_dim, intermediate_dim)
        self.up_proj = GroupedLinear(num_experts, hidden_dim, intermediate_dim)
        self.down_proj = GroupedLinear(num_experts, intermediate_dim, hidden_dim)

    def forward(
        self,
        permuted_x: torch.Tensor,
        permuted_probs: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Computes expert outputs for sorted input tokens.

        Args:
            permuted_x: Input tokens sorted by their assigned expert. Shape: ``(num_tokens, hidden_size)``.
            permuted_probs: Routing probabilities of the sorted tokens. Shape: ``(num_tokens,)``.
            tokens_per_expert: CPU tensor with the number of tokens assigned to each expert, in order.
                Shape: ``(num_experts,)``.

        Returns:
            The expert outputs weighted by the routing probabilities, in the sorted order of ``permuted_x``.
            Shape: ``(num_tokens, hidden_size)``.
        """
        # No tokens were routed to the experts of this rank.
        if permuted_x.numel() == 0:
            return permuted_x

        probs = permuted_probs[:, None].to(permuted_x.dtype)
        values = self.down_proj(
            silu_mul(self.gate_proj(permuted_x, tokens_per_expert), self.up_proj(permuted_x, tokens_per_expert)),
            tokens_per_expert,
        )

        return probs * values

    def reset_parameters(self):
        """Resets module parameters."""
        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()
