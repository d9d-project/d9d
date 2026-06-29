import torch
from torch import nn

from d9d.module.base import ModuleLateInit


class LightningIndexer(nn.Module, ModuleLateInit):
    """Implements the lightning indexer of DeepSeek Sparse Attention (DSA).

    The lightning indexer is a lightweight scorer that decides, for every query
    token, which preceding tokens are worth attending to. For a query token
    ``h_t`` and a preceding token ``h_s`` the index score is a gated sum of
    per-head dot products (Eq. 1 of the DeepSeek-V3.2 report):

        I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} . k_s)

    where ``j`` indexes the (few) indexer heads, ``q_{t,j}`` and the scalar gate
    ``w_{t,j}`` are derived from the query token, and ``k_s`` is a single key
    vector shared across all indexer heads (MQA style). ReLU is chosen for
    throughput; with a small head count the indexer is far cheaper than the main
    attention even though it remains O(L^2).

    The downstream fine-grained token selection retrieves only the key-value
    entries whose index score lies in the top-k of ``I_{t,:}``. This module owns
    the scoring and the selection; the additive mask it produces plugs directly
    into a standard scaled-dot-product attention backend.

    References:
        [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, top_k: int) -> None:
        """Constructs the LightningIndexer.

        Args:
            hidden_size: Model hidden dimension that queries and keys are derived from.
            num_heads: Number of indexer heads (``H_I``). Kept small for efficiency.
            head_dim: Per-head dimension of the indexer queries and keys (``d_I``).
            top_k: Number of preceding tokens selected per query. When a sequence is
                shorter than ``top_k`` the selection degrades gracefully to dense attention.
        """
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._top_k = top_k
        self._logit_scale = head_dim**-0.5
        self._weight_scale = num_heads**-0.5

        # Per-head indexer queries, a single shared key, and per-head scalar gates.
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)

    @property
    def top_k(self) -> int:
        """Number of preceding tokens each query selects."""
        return self._top_k

    def index_scores(self, hidden_states: torch.Tensor, attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Computes the index score between every query and every key token.

        The scores are accumulated in float32 regardless of the input dtype so that
        the subsequent top-k ranking is not perturbed by low-precision rounding.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_bias: Optional additive bias broadcast onto the scores, e.g. a
                causal bias holding ``-inf`` at disallowed positions so they are never
                selected. Shape broadcastable to ``(batch, seq_len, seq_len)``.

        Returns:
            Index scores. Shape: ``(batch, seq_len_q, seq_len_k)``.
        """
        q = self.q_proj(hidden_states).float().view(*hidden_states.shape[:-1], self._num_heads, self._head_dim)
        k = self.k_proj(hidden_states).float()
        weights = self.weights_proj(hidden_states).float() * self._weight_scale

        # Per-head ReLU(q * k), then gate and sum over the indexer heads.
        per_head = torch.relu(torch.einsum("bqhd,bkd->bqhk", q, k) * self._logit_scale)
        scores = torch.einsum("bqh,bqhk->bqk", weights, per_head)

        if attention_bias is not None:
            scores = scores + attention_bias

        return scores

    def select_top_k(self, hidden_states: torch.Tensor, attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Selects the indices of the top-k scoring key tokens for each query.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_bias: Optional additive bias applied before ranking (see ``index_scores``).

        Returns:
            Indices of the selected key tokens. Shape: ``(batch, seq_len_q, k)`` where
            ``k = min(top_k, seq_len_k)``.
        """
        scores = self.index_scores(hidden_states, attention_bias)
        k = min(self._top_k, scores.shape[-1])
        return scores.topk(k, dim=-1).indices

    def forward(self, hidden_states: torch.Tensor, attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Builds the additive selection mask for the fine-grained token selection.

        The mask holds ``0`` at the top-k selected positions and ``-inf`` everywhere
        else, ready to be added to attention logits before the softmax.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_bias: Optional additive bias applied before ranking (see ``index_scores``).

        Returns:
            Additive selection mask. Shape: ``(batch, seq_len_q, seq_len_k)``.
        """
        top_k_indices = self.select_top_k(hidden_states, attention_bias)
        # Queries and keys are both derived from hidden_states, so the mask is square.
        batch, seq_len, _ = hidden_states.shape
        mask = hidden_states.new_full((batch, seq_len, seq_len), float("-inf"))
        return mask.scatter_(-1, top_k_indices, 0.0)

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.weights_proj.reset_parameters()


def _build_causal_bias(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Builds an additive causal bias of shape ``(seq_len, seq_len)``.

    Entry ``(q, k)`` is ``0`` when key ``k`` may be attended by query ``q`` (``k <= q``)
    and ``-inf`` otherwise.

    Returns:
        The additive causal bias tensor.
    """
    positions = torch.arange(seq_len, device=device)
    disallowed = positions.unsqueeze(0) > positions.unsqueeze(1)
    return torch.zeros(seq_len, seq_len, device=device, dtype=dtype).masked_fill_(disallowed, float("-inf"))


def build_sparse_selection_mask(
    indexer: LightningIndexer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Builds the additive causal + top-k selection mask for sparse causal attention.

    Tokens are ranked under a causal bias (so future positions are never preferred); only
    the indexer's top-k per query survive, and causality is enforced again on the returned
    mask (top-k may still include future slots when ``k`` exceeds the causal context). The
    result is added to the attention logits before the softmax, which is what realises the
    fine-grained token selection of DSA on top of a dense attention backend.

    Args:
        indexer: The lightning indexer producing the per-query token selection.
        hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
        attention_mask: Optional additive mask (e.g. padding) added on top of the causal and
            selection masks. Broadcastable to ``(batch, 1, seq_len, seq_len)``.

    Returns:
        Additive attention mask. Shape: ``(batch, 1, seq_len, seq_len)``.
    """
    seq_len = hidden_states.shape[1]
    causal_bias = _build_causal_bias(seq_len, hidden_states.device, hidden_states.dtype)
    selection_mask = indexer(hidden_states, attention_bias=causal_bias)
    sparse_mask = (selection_mask + causal_bias).unsqueeze(1)
    if attention_mask is not None:
        sparse_mask = sparse_mask + attention_mask
    return sparse_mask
