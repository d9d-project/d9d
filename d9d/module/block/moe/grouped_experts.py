import torch
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from torch import nn

from d9d.module.base import ModuleLateInit

from .grouped_linear import GroupedLinear

# TODO: migrate fused SiLU from liger kernel to custom impl (or use those written by SGL team)


class GroupedSwiGLU(nn.Module, ModuleLateInit):
    """
    Executes a collection of SwiGLU experts efficiently using Grouped GEMM.

    This module implements the architectural pattern: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`.
    It applies this operation across multiple discrete experts in parallel without padding or masking.
    """

    def __init__(
            self,
            hidden_dim: int,
            intermediate_dim: int,
            num_experts: int
    ):
        """
        Constructs the GroupedSwiGLU module.

        Args:
            hidden_dim: Dimensionality of the input and output hidden states.
            intermediate_dim: Dimensionality of the intermediate projection.
            num_experts: Total number of experts managed by this local instance.
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
        """
        Computes expert outputs for sorted input tokens.

        Args:
            permuted_x: Input tokens sorted by their assigned expert.
                Shape: `(total_tokens, hidden_dim)`.
            permuted_probs: Routing weights/probabilities corresponding to the sorted tokens.
                Shape: `(total_tokens)`.
            tokens_per_expert: Number of tokens assigned to each consecutive expert. It is a CPU tensor.
                Shape: `(num_experts)`.

        Returns:
            The computed and weighted output tokens (still permuted).
            Shape: `(total_tokens, hidden_dim)`.
        """

        if permuted_x.numel() == 0:  # handle cases when there are no routed experts to this instance
            return permuted_x

        probs = permuted_probs[:, None].to(permuted_x.dtype)
        values = self.down_proj(
            LigerSiLUMulFunction.apply(
                self.gate_proj(permuted_x, tokens_per_expert),
                self.up_proj(permuted_x, tokens_per_expert)
            ),
            tokens_per_expert
        )

        return probs * values

    def reset_parameters(self):
        """Resets parameters for all internal linear projections."""

        self.gate_proj.reset_parameters()
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()
