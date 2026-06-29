import pytest
import torch
from d9d.module.block.attention import LightningIndexer
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


def _build_indexer(top_k: int) -> LightningIndexer:
    torch.manual_seed(42)
    indexer = LightningIndexer(
        hidden_size=_HIDDEN,
        num_heads=_INDEX_HEADS,
        head_dim=_INDEX_HEAD_DIM,
        top_k=top_k,
    )
    indexer.reset_parameters()
    return indexer


@pytest.mark.local
def test_index_scores_match_equation() -> None:
    """index_scores must equal the gated ReLU dot-product of Eq. 1."""
    indexer = _build_indexer(top_k=4)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN)

    actual = indexer.index_scores(x)

    q = indexer.q_proj(x).view(_BATCH, _SEQ, _INDEX_HEADS, _INDEX_HEAD_DIM)
    k = indexer.k_proj(x)
    weights = indexer.weights_proj(x) * (_INDEX_HEADS**-0.5)
    per_head = torch.relu(torch.einsum("bqhd,bkd->bqhk", q, k) * (_INDEX_HEAD_DIM**-0.5))
    expected = torch.einsum("bqh,bqhk->bqk", weights, per_head)

    assert_close(actual, expected)


@pytest.mark.local
def test_selection_mask_matches_top_k_indices() -> None:
    """The additive mask is zero exactly at the selected indices and -inf elsewhere."""
    top_k = 4
    indexer = _build_indexer(top_k=top_k)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN)
    bias = _causal_bias(_SEQ)

    indices = indexer.select_top_k(x, attention_bias=bias)
    mask = indexer(x, attention_bias=bias)

    assert indices.shape == (_BATCH, _SEQ, min(top_k, _SEQ))
    rebuilt = torch.full((_BATCH, _SEQ, _SEQ), float("-inf")).scatter_(-1, indices, 0.0)
    assert_close(mask, rebuilt)


@pytest.mark.local
def test_causal_bias_excludes_future() -> None:
    """With a causal bias, no future token is ever selected and every query keeps at least one."""
    indexer = _build_indexer(top_k=3)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN)
    bias = _causal_bias(_SEQ)

    mask = indexer(x, attention_bias=bias) + bias
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

    mask = indexer(x, attention_bias=bias) + bias
    counts = (mask > float("-inf")).sum(dim=-1)

    causal_context = torch.arange(1, _SEQ + 1)
    assert (counts <= min(top_k, _SEQ)).all()
    assert (counts <= causal_context).all()


@pytest.mark.local
def test_index_scores_are_float32() -> None:
    """Scores are accumulated in float32 even for low-precision inputs to stabilise top-k."""
    indexer = _build_indexer(top_k=4).to(torch.bfloat16)
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, dtype=torch.bfloat16)
    assert indexer.index_scores(x).dtype == torch.float32
