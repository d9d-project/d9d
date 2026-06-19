from __future__ import annotations

import typing

import torch
from torch import nn
from torch.distributed import ProcessGroup

from d9d.module.base import ModuleLateInit

from .communications import (
    ExpertCommunicationHandler,
    NoCommunicationHandler,
)
from .grouped_experts import GroupedSwiGLU
from .router import RoutingResult, TopKRouter
from .shared_expert import SharedExpertParameters, SharedSwiGLU

if typing.TYPE_CHECKING:
    from .replay import RouterReplayRecorder


class MoELayer(nn.Module, ModuleLateInit):
    """A complete Mixture-of-Experts (MoE) block comprising routing, communication, and computation.

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
        router_renormalize_probabilities: bool,
        shared_expert: SharedExpertParameters | None = None,
    ):
        """Constructs the MoELayer.

        Args:
            hidden_dim: Hidden size.
            intermediate_dim_grouped: Intermediate dimension for the Expert FFNs.
            num_grouped_experts: Total number of experts.
            top_k: Number of experts to route each token to.
            router_renormalize_probabilities: Configures router probability normalization behavior.
            shared_expert: Optional configuration for a shared expert.
        """
        super().__init__()
        self.router = TopKRouter(
            dim=hidden_dim,
            num_experts=num_grouped_experts,
            top_k=top_k,
            renormalize_probabilities=router_renormalize_probabilities,
        )
        self.grouped_experts = GroupedSwiGLU(
            hidden_dim=hidden_dim, intermediate_dim=intermediate_dim_grouped, num_experts=num_grouped_experts
        )
        self._communicator: ExpertCommunicationHandler = NoCommunicationHandler(num_grouped_experts)

        if shared_expert is not None:
            self.shared_expert = SharedSwiGLU(hidden_size=hidden_dim, params=shared_expert)
        else:
            self.shared_expert = None

        self._num_grouped_experts = num_grouped_experts
        self._hidden_dim = hidden_dim

        self.tokens_per_expert = nn.Buffer(torch.empty((num_grouped_experts,), dtype=torch.int64), persistent=False)

        # Set by a RouterReplayRecorder while it is installed; used to capture this layer's expert selection.
        self._replay_recorder: RouterReplayRecorder | None = None
        self._replay_layer_id: int | None = None

    def bind_replay_recorder(self, recorder: RouterReplayRecorder, layer_id: int) -> None:
        """Attaches a routing recorder to this layer so non-replay forwards capture their expert selection.

        Args:
            recorder: The recorder that will receive this layer's selection.
            layer_id: Stable identifier of this layer within the recorded model.
        """
        self._replay_recorder = recorder
        self._replay_layer_id = layer_id

    def unbind_replay_recorder(self) -> None:
        """Detaches any routing recorder previously attached via `bind_replay_recorder`."""
        self._replay_recorder = None
        self._replay_layer_id = None

    def enable_distributed_communicator(self, group: ProcessGroup):
        """Switches from local no-op communication to distributed DeepEP communication.

        This should be called during model initialization if the model is running in a
        distributed Expert Parallel environment.

        Args:
            group: The PyTorch process group spanning the expert parallel ranks.
        """
        # Lazy load the handler to prevent early DeepEP bindings/evaluation
        from .communications.deepep import DeepEpCommunicationHandler  # noqa: PLC0415

        communicator = DeepEpCommunicationHandler(num_experts=self._num_grouped_experts)
        communicator.setup(group, self._hidden_dim, self.router.gate.weight.dtype)
        self._communicator = communicator

    @torch.no_grad()
    def _update_tokens_per_expert(self, expert_indices: torch.Tensor):
        flat_indices = expert_indices.view(-1)
        ones = torch.ones_like(flat_indices, dtype=self.tokens_per_expert.dtype)
        self.tokens_per_expert.scatter_add_(0, flat_indices, ones)

    @torch.no_grad()
    def reset_stats(self):
        """Resets the expert load balancing counters."""
        self.tokens_per_expert.zero_()

    def forward(self, hidden_states: torch.Tensor, replay_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Routes tokens to experts, computes, and combines results.

        Args:
            hidden_states: Input tensor. Shape: `(batch_size, seq_len, hidden_dim)`.
            replay_indices: Optional recorded expert indices to replay instead of recomputing the routing selection
                (Expert Replay). Shape: `(batch_size, seq_len, top_k)`. When provided, recording is skipped.

        Returns:
            Output tensor combined from experts. Shape: `(batch_size, seq_len, hidden_dim)`.
        """
        old_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        if replay_indices is not None:
            replay_indices = replay_indices.reshape(-1, replay_indices.shape[-1])

        if self.shared_expert is not None:
            shared_expert_result = self.shared_expert(hidden_states)
        else:
            shared_expert_result = None

        routing_result: RoutingResult = self.router(hidden_states, replay_indices=replay_indices)
        expert_indices = routing_result.selected_expert_indices
        expert_scores = routing_result.selected_probabilities
        self._update_tokens_per_expert(expert_indices)

        if self._replay_recorder is not None and replay_indices is None:
            assert self._replay_layer_id is not None  # noqa: S101 -- always set together with the recorder
            self._replay_recorder.capture(self._replay_layer_id, expert_indices.reshape(*old_shape[:-1], -1))
        hidden_states, expert_scores, expert_count = self._communicator.dispatch(
            hidden_states, expert_indices, expert_scores
        )
        hidden_states = self.grouped_experts(hidden_states, expert_scores, expert_count)
        hidden_states = self._communicator.combine(hidden_states)

        if shared_expert_result is not None:
            hidden_states = hidden_states + shared_expert_result

        hidden_states = hidden_states.reshape(*old_shape)

        return hidden_states

    def reset_parameters(self):
        """Resets module parameters."""
        self.router.reset_parameters()
        self.grouped_experts.reset_parameters()
        if self.shared_expert is not None:
            self.shared_expert.reset_parameters()

        nn.init.zeros_(self.tokens_per_expert)
