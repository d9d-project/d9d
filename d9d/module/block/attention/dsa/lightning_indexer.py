import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.positional import RotaryEmbeddingApplicator, RotaryEmbeddingStyle


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

    As in the reference implementation, rotary embeddings are applied to the leading
    ``rope_dim`` components of the indexer queries and keys, so the index scores are
    position-aware and reuse the ``(cos, sin)`` embeddings of the main attention.

    The downstream fine-grained token selection retrieves only the key-value
    entries whose index score lies in the top-k of ``I_{t,:}``. This module owns
    the scoring and the selection; the additive mask it produces plugs directly
    into a standard scaled-dot-product attention backend.

    References:
        [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        top_k: int,
        rope_style: RotaryEmbeddingStyle,
        rope_dim: int | None = None,
    ) -> None:
        """Constructs the LightningIndexer.

        Args:
            hidden_size: Model hidden dimension that queries and keys are derived from.
            num_heads: Number of indexer heads (``H_I``). Kept small for efficiency.
            head_dim: Per-head dimension of the indexer queries and keys (``d_I``).
            top_k: Number of preceding tokens selected per query. When a sequence is
                shorter than ``top_k`` the selection degrades gracefully to dense attention.
            rope_style: Rotary embedding layout style alignment.
            rope_dim: Dimension of the RoPE sub-vector of an indexer head. It must match the
                dimension of the ``(cos, sin)`` embeddings the main attention is driven with.
                If ``None``, RoPE is applied to the full ``head_dim``.

        Raises:
            ValueError: If ``rope_dim`` exceeds ``head_dim``.
        """
        super().__init__()
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._top_k = top_k
        self._logit_scale = head_dim**-0.5
        self._weight_scale = num_heads**-0.5

        self._rope_dim = rope_dim if rope_dim is not None else head_dim
        self._nope_dim = head_dim - self._rope_dim

        if self._nope_dim < 0:
            raise ValueError(f"Indexer rope_dim ({self._rope_dim}) must not exceed head_dim ({head_dim}).")

        # Per-head indexer queries, a single shared key, and per-head scalar gates.
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.weights_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.rope = RotaryEmbeddingApplicator(style=rope_style)

    @property
    def top_k(self) -> int:
        """Number of preceding tokens each query selects."""
        return self._top_k

    def _apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotates the leading ``rope_dim`` components of the indexer queries and keys.

        Returns:
            A tuple of the rotated query and key states.
        """
        cos, sin = position_embeddings

        if self._nope_dim == 0:
            return self.rope(query_states, key_states, cos, sin)

        q_rope, q_nope = query_states.split([self._rope_dim, self._nope_dim], dim=-1)
        k_rope, k_nope = key_states.split([self._rope_dim, self._nope_dim], dim=-1)
        q_rope, k_rope = self.rope(q_rope, k_rope, cos, sin)
        return torch.cat([q_rope, q_nope], dim=-1), torch.cat([k_rope, k_nope], dim=-1)

    def index_scores(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes the index score between every query and every key token.

        The scores are accumulated in float32 regardless of the input dtype so that
        the subsequent top-k ranking is not perturbed by low-precision rounding.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_bias: Optional additive bias broadcast onto the scores, e.g. a
                causal bias holding ``-inf`` at disallowed positions so they are never
                selected. Shape broadcastable to ``(batch, seq_len, seq_len)``.
            position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application.
                Each tensor shape: ``(batch, seq_len, rope_dim)``.

        Returns:
            Index scores. Shape: ``(batch, seq_len_q, seq_len_k)``.
        """
        q = self.q_proj(hidden_states).float().view(*hidden_states.shape[:-1], self._num_heads, self._head_dim)
        # The single indexer key is shared across heads; the head axis is kept for RoPE and dropped after.
        k = self.k_proj(hidden_states).float().unsqueeze(-2)
        q, k = self._apply_rope(q, k, position_embeddings)
        k = k.squeeze(-2)

        weights = self.weights_proj(hidden_states).float() * self._weight_scale

        # Per-head ReLU(q * k), then gate and sum over the indexer heads.
        per_head = torch.relu(torch.einsum("bqhd,bkd->bqhk", q, k) * self._logit_scale)
        scores = torch.einsum("bqh,bqhk->bqk", weights, per_head)

        if attention_bias is not None:
            scores = scores + attention_bias

        return scores

    def select_top_k(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Selects the indices of the top-k scoring key tokens for each query.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_bias: Optional additive bias applied before ranking (see ``index_scores``).
            position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application.

        Returns:
            Indices of the selected key tokens. Shape: ``(batch, seq_len_q, k)`` where
            ``k = min(top_k, seq_len_k)``.
        """
        scores = self.index_scores(hidden_states, attention_bias, position_embeddings)
        k = min(self._top_k, scores.shape[-1])
        return scores.topk(k, dim=-1).indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Builds the additive selection mask for the fine-grained token selection.

        The mask holds ``0`` at the top-k selected positions and ``-inf`` everywhere
        else, ready to be added to attention logits before the softmax.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_bias: Optional additive bias applied before ranking (see ``index_scores``).
            position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application.

        Returns:
            Additive selection mask. Shape: ``(batch, seq_len_q, seq_len_k)``.
        """
        top_k_indices = self.select_top_k(hidden_states, attention_bias, position_embeddings)
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
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
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
        position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application.

    Returns:
        Additive attention mask. Shape: ``(batch, 1, seq_len, seq_len)``.
    """
    seq_len = hidden_states.shape[1]
    causal_bias = _build_causal_bias(seq_len, hidden_states.device, hidden_states.dtype)
    selection_mask = indexer(hidden_states, causal_bias, position_embeddings)
    sparse_mask = (selection_mask + causal_bias).unsqueeze(1)
    if attention_mask is not None:
        sparse_mask = sparse_mask + attention_mask
    return sparse_mask


_TARGET_PROBABILITY_BUDGET = 1 << 26
"""
Upper bound on the number of attention probabilities materialised at once when building the KL target.

The target needs the full ``(batch, num_heads, seq_len, seq_len)`` attention distribution, which is
larger than the ``(batch, seq_len, seq_len)`` index scores by a factor of the head count; queries are
therefore processed in chunks sized so that this budget is respected.
"""


@torch.no_grad()
def _head_averaged_attention_probabilities(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_scale: float,
    attention_bias: torch.Tensor,
) -> torch.Tensor:
    """Computes the head-averaged attention distribution that the indexer is trained against.

    DSA aligns the indexer with the sum of the per-head attention distributions, L1-normalised;
    since every head contributes a distribution summing to one, that normalisation is exactly the
    average over heads. The target is a constant of the indexer objective, so it is computed
    without autograd and in query chunks, which keeps peak memory at ``_TARGET_PROBABILITY_BUDGET``
    and makes the objective independent of the SDPA backend used by the main attention.

    Args:
        query_states: Rotated query states. Shape: ``(batch, seq_len, n_q_heads, head_dim)``.
        key_states: Rotated key states. Shape: ``(batch, seq_len, n_kv_heads, head_dim)``.
        attention_scale: Softmax scaling factor of the attention the target is taken from.
        attention_bias: Additive bias (causal, padding) applied to the attention logits.
            Broadcastable to ``(batch, 1, seq_len, seq_len)``.

    Returns:
        Target distribution over the key tokens. Shape: ``(batch, seq_len_q, seq_len_k)``.
    """
    batch, seq_len, num_q_heads, _ = query_states.shape
    groups = num_q_heads // key_states.shape[2]

    # (B, S, H, D) -> (B, H, S, D)
    query = query_states.transpose(1, 2).float()
    key = key_states.transpose(1, 2).repeat_interleave(groups, dim=1).float()

    chunk = max(1, _TARGET_PROBABILITY_BUDGET // (batch * num_q_heads * seq_len))
    probabilities = []

    for start in range(0, seq_len, chunk):
        logits = torch.matmul(query[:, :, start : start + chunk], key.transpose(2, 3)) * attention_scale
        logits = logits + attention_bias[..., start : start + chunk, :]
        probabilities.append(torch.softmax(logits, dim=-1).sum(dim=1))

    return torch.cat(probabilities, dim=1) / num_q_heads


def indexer_kl_loss(
    indexer: LightningIndexer,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_scale: float,
    attention_mask: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Computes the auxiliary objective that trains the lightning indexer.

    The top-k selection consumed by the attention is non-differentiable, so the main loss carries no
    signal for the indexer. DSA therefore trains it to imitate the attention it gates: with
    ``p_{t,:}`` the head-averaged attention distribution of the main attention, the indexer minimises

        L_I = sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))

    over the causally allowed key positions. The target is a constant that is recomputed here from the
    query and key states rather than read out of the attention kernel, which keeps every SDPA backend
    (including the FlashAttention ones, which only return the attention output) usable for the main
    path. As in DeepSeek-V3.2 the indexer is optimized separately from the main model, so
    ``hidden_states`` should be detached by the caller if the main model is trained concurrently.

    Args:
        indexer: The lightning indexer to train.
        hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
        query_states: Rotated query states of the main attention. Shape:
            ``(batch, seq_len, n_q_heads, head_dim)``.
        key_states: Rotated key states of the main attention. Shape:
            ``(batch, seq_len, n_kv_heads, head_dim)``.
        attention_scale: Softmax scaling factor of the main attention.
        attention_mask: Optional additive mask (e.g. padding) applied to both distributions.
            Broadcastable to ``(batch, 1, seq_len, seq_len)``.
        position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application.

    Returns:
        The KL objective, averaged over batch elements and query positions. Shape: ``()``.
    """
    batch, seq_len, _ = hidden_states.shape
    bias = _build_causal_bias(seq_len, hidden_states.device, torch.float32)

    if attention_mask is not None:
        bias = bias + attention_mask

    target = _head_averaged_attention_probabilities(query_states, key_states, attention_scale, bias)

    index_bias = bias.expand(batch, 1, seq_len, seq_len).squeeze(1)
    log_probabilities = torch.log_softmax(indexer.index_scores(hidden_states, index_bias, position_embeddings), dim=-1)

    # Disallowed positions hold zero target mass against a ``-inf`` log-probability; their
    # contribution is zero by convention but evaluates to NaN, hence the explicit selection.
    contributions = torch.where(target > 0.0, target * (torch.log(target) - log_probabilities), 0.0)
    return contributions.sum(dim=-1).mean()
