import pytest
import torch
from d9d.module.block.attention import LightningIndexer
from d9d.module.block.positional import RotaryEmbeddingStyle
from d9d.module.block.positional.rope import prepare_rotary_cos_sin_emb
from torch.testing import assert_close

_HIDDEN = 32
_INDEX_HEADS = 3
_INDEX_HEAD_DIM = 8
_BATCH = 2
_SEQ = 12


def _causal_bias(seq_len: int) -> torch.Tensor:
    positions = torch.arange(seq_len)
    disallowed = positions.unsqueeze(0) > positions.unsqueeze(1)
    return torch.zeros(seq_len, seq_len).masked_fill_(disallowed, float("-inf"))


def _position_embeddings(rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = prepare_rotary_cos_sin_emb(
        rope_base=10000,
        head_dim=rope_dim,
        max_position_ids=_SEQ,
        device=torch.device("cpu"),
        dtype=torch.float32,
        style=RotaryEmbeddingStyle.HALF,
    )
    return cos.unsqueeze(0).expand(_BATCH, -1, -1), sin.unsqueeze(0).expand(_BATCH, -1, -1)


def _build_indexer(top_k: int, rope_dim: int | None = None) -> LightningIndexer:
    torch.manual_seed(42)
    indexer = LightningIndexer(
        hidden_size=_HIDDEN,
        num_heads=_INDEX_HEADS,
        head_dim=_INDEX_HEAD_DIM,
        top_k=top_k,
        rope_style=RotaryEmbeddingStyle.HALF,
        rope_dim=rope_dim,
    )
    indexer.reset_parameters()
    return indexer


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rope(x: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Independent HALF-style RoPE for tensors shaped ``(batch, seq_len, heads, dim)``."""
    cos, sin = position_embeddings
    return x * cos.unsqueeze(2) + _rotate_half(x) * sin.unsqueeze(2)


def _reference_index_scores(
    indexer: LightningIndexer,
    x: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    rope_dim: int,
) -> torch.Tensor:
    """Eq. 1 of the DeepSeek-V3.2 report, with RoPE on the leading ``rope_dim`` components."""
    q = indexer.q_proj(x).view(_BATCH, _SEQ, _INDEX_HEADS, _INDEX_HEAD_DIM)
    k = indexer.k_proj(x).unsqueeze(2)

    q_rope, q_nope = q.split([rope_dim, _INDEX_HEAD_DIM - rope_dim], dim=-1)
    k_rope, k_nope = k.split([rope_dim, _INDEX_HEAD_DIM - rope_dim], dim=-1)
    q = torch.cat([_rope(q_rope, position_embeddings), q_nope], dim=-1)
    k = torch.cat([_rope(k_rope, position_embeddings), k_nope], dim=-1).squeeze(2)

    weights = indexer.weights_proj(x) * (_INDEX_HEADS**-0.5)
    per_head = torch.relu(torch.einsum("bqhd,bkd->bqhk", q, k) * (_INDEX_HEAD_DIM**-0.5))
    return torch.einsum("bqh,bqhk->bqk", weights, per_head)


@pytest.mark.local
@pytest.mark.parametrize("rope_dim", [None, 4])
def test_index_scores_match_equation(rope_dim: int | None) -> None:
    """index_scores must equal the gated ReLU dot-product of Eq. 1 over RoPE-rotated q/k."""
    indexer = _build_indexer(top_k=4, rope_dim=rope_dim)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN)
    effective_rope_dim = rope_dim if rope_dim is not None else _INDEX_HEAD_DIM
    rope = _position_embeddings(effective_rope_dim)

    actual = indexer.index_scores(x, None, rope)
    expected = _reference_index_scores(indexer, x, rope, effective_rope_dim)

    assert_close(actual, expected)


@pytest.mark.local
def test_index_scores_depend_on_position() -> None:
    """RoPE makes the scores position-aware: identical tokens at different offsets score differently."""
    indexer = _build_indexer(top_k=4)
    rope = _position_embeddings(_INDEX_HEAD_DIM)

    x = torch.randn(_BATCH, 1, _HIDDEN).expand(_BATCH, _SEQ, _HIDDEN).contiguous()
    scores = indexer.index_scores(x, None, rope)

    # Every query sees the very same token at every position, so without RoPE each row would be constant.
    assert not torch.allclose(scores[:, -1, 0], scores[:, -1, -1])


@pytest.mark.local
def test_rope_dim_exceeding_head_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="rope_dim"):
        LightningIndexer(
            hidden_size=_HIDDEN,
            num_heads=_INDEX_HEADS,
            head_dim=_INDEX_HEAD_DIM,
            top_k=4,
            rope_style=RotaryEmbeddingStyle.HALF,
            rope_dim=_INDEX_HEAD_DIM + 2,
        )


@pytest.mark.local
def test_selection_mask_matches_top_k_indices() -> None:
    """The additive mask is zero exactly at the selected indices and -inf elsewhere."""
    top_k = 4
    indexer = _build_indexer(top_k=top_k)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN)
    bias = _causal_bias(_SEQ)
    rope = _position_embeddings(_INDEX_HEAD_DIM)

    indices = indexer.select_top_k(x, bias, rope)
    mask = indexer(x, bias, rope)

    assert indices.shape == (_BATCH, _SEQ, min(top_k, _SEQ))
    rebuilt = torch.full((_BATCH, _SEQ, _SEQ), float("-inf")).scatter_(-1, indices, 0.0)
    assert_close(mask, rebuilt)


@pytest.mark.local
def test_causal_bias_excludes_future() -> None:
    """With a causal bias, no future token is ever selected and every query keeps at least one."""
    indexer = _build_indexer(top_k=3)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN)
    bias = _causal_bias(_SEQ)

    mask = indexer(x, bias, _position_embeddings(_INDEX_HEAD_DIM)) + bias
    allowed = mask > float("-inf")

    future = torch.triu(torch.ones(_SEQ, _SEQ, dtype=torch.bool), diagonal=1)
    assert not (allowed & future).any()
    assert (allowed.sum(dim=-1) >= 1).all()


@pytest.mark.local
@pytest.mark.parametrize("top_k", [1, 4, _SEQ + 8])
def test_attended_count_never_exceeds_top_k(top_k: int) -> None:
    """Each query attends to at most ``top_k`` (and at most the causal context) tokens."""
    indexer = _build_indexer(top_k=top_k)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN)
    bias = _causal_bias(_SEQ)

    mask = indexer(x, bias, _position_embeddings(_INDEX_HEAD_DIM)) + bias
    counts = (mask > float("-inf")).sum(dim=-1)

    causal_context = torch.arange(1, _SEQ + 1)
    assert (counts <= min(top_k, _SEQ)).all()
    assert (counts <= causal_context).all()


@pytest.mark.local
def test_index_scores_are_float32() -> None:
    """Scores are accumulated in float32 even for low-precision inputs to stabilise top-k."""
    indexer = _build_indexer(top_k=4).to(torch.bfloat16)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, dtype=torch.bfloat16)
    cos, sin = _position_embeddings(_INDEX_HEAD_DIM)
    scores = indexer.index_scores(x, None, (cos.bfloat16(), sin.bfloat16()))
    assert scores.dtype == torch.float32
