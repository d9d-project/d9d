import torch
from torch import nn
from torch.distributed import ProcessGroup

from d9d.module.base import ModuleLateInit

from .communications import (
    DeepEpCommunicationHandler,
    ExpertCommunicationHandler,
    NoCommunicationHandler,
)
from .grouped_experts import GroupedSwiGLU
from .router import TopKRouter

# TODO: implement expert bias
# TODO: shared experts


class MoELayer(nn.Module, ModuleLateInit):
    """
    A complete Mixture-of-Experts (MoE) block comprising routing, communication, and computation.

    This layer integrates:

    1.  **Router**: Selects experts for each token.
    2.  **Communicator**: Handles token dispatch to local or remote experts (EP).
    3.  **Experts**: Performs parallelized computation (Grouped SwiGLU).
    """

    def __init__(
            self,
            hidden_dim: int,
            intermediate_dim_grouped: int,
            num_grouped_experts: int,
            top_k: int,
            router_renormalize_probabilities: bool
    ):
        """
        Constructs the MoELayer.

       Args:
           hidden_dim: Hidden size.
           intermediate_dim_grouped: Intermediate dimension for the Expert FFNs.
           num_grouped_experts: Total number of experts.
           top_k: Number of experts to route each token to.
           router_renormalize_probabilities: Configures router probability normalization behavior.
       """

        super().__init__()
        self.router = TopKRouter(
            dim=hidden_dim, num_experts=num_grouped_experts, top_k=top_k,
            renormalize_probabilities=router_renormalize_probabilities
        )
        self.grouped_experts = GroupedSwiGLU(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim_grouped,
            num_experts=num_grouped_experts
        )
        self._communicator: ExpertCommunicationHandler = NoCommunicationHandler(num_grouped_experts)

        self._num_grouped_experts = num_grouped_experts
        self._hidden_dim = hidden_dim

        self.tokens_per_expert = nn.Buffer(torch.empty((num_grouped_experts,), dtype=torch.int64), persistent=False)

    def enable_distributed_communicator(self, group: ProcessGroup):
        """
        Switches from local no-op communication to distributed DeepEP communication.

        This should be called during model initialization if the model is running in a
        distributed Expert Parallel environment.

        Args:
            group: The PyTorch process group spanning the expert parallel ranks.
        """

        communicator = DeepEpCommunicationHandler(num_experts=self._num_grouped_experts)
        communicator.setup(group, self._hidden_dim, self.router.gate.weight.dtype)
        self._communicator = communicator

    @torch.no_grad()
    def _update_tokens_per_expert(self, expert_indices: torch.Tensor):
        self.tokens_per_expert.add_(expert_indices.view(-1).bincount(minlength=self._num_grouped_experts))

    @torch.no_grad()
    def reset_stats(self):
        """Resets the expert load balancing counters."""
        self.tokens_per_expert.zero_()

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Routes tokens to experts, computes, and combines results.

        Args:
            hidden_states: Input tensor. Shape: `(batch_size, seq_len, hidden_dim)`.

        Returns:
            Output tensor combined from experts. Shape: `(batch_size, seq_len, hidden_dim)`.
        """

        old_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        expert_indices, expert_scores = self.router(hidden_states)
        self._update_tokens_per_expert(expert_indices)
        hidden_states, expert_scores, expert_count = self._communicator.dispatch(
            hidden_states, expert_indices, expert_scores
        )
        hidden_states = self.grouped_experts(hidden_states, expert_scores, expert_count)
        hidden_states = self._communicator.combine(hidden_states)
        hidden_states = hidden_states.reshape(*old_shape)

        return hidden_states

    def reset_parameters(self):
        """Resets module parameters."""
        self.router.reset_parameters()
        self.grouped_experts.reset_parameters()

        nn.init.zeros_(self.tokens_per_expert)
