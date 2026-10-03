import pytest
import torch
from d9d.module.block.embedding import merge_media_embeddings


@pytest.mark.local
def test_merge_replaces_masked_positions():
    token_embeddings = torch.zeros(2, 4, 3)
    media_token_mask = torch.tensor([[False, True, True, False], [True, False, False, False]])
    media_embeddings = torch.arange(9, dtype=torch.float32).view(3, 3)

    merged = merge_media_embeddings(token_embeddings, media_token_mask, media_embeddings)

    torch.testing.assert_close(merged[0, 1], media_embeddings[0])
    torch.testing.assert_close(merged[0, 2], media_embeddings[1])
    torch.testing.assert_close(merged[1, 0], media_embeddings[2])
    torch.testing.assert_close(merged[0, 0], torch.zeros(3))
    torch.testing.assert_close(merged[1, 1:], torch.zeros(3, 3))


@pytest.mark.local
def test_merge_empty_mask_keeps_embeddings():
    token_embeddings = torch.randn(2, 4, 3)
    media_token_mask = torch.zeros(2, 4, dtype=torch.bool)
    media_embeddings = torch.empty(0, 3)

    merged = merge_media_embeddings(token_embeddings, media_token_mask, media_embeddings)

    torch.testing.assert_close(merged, token_embeddings)


@pytest.mark.local
def test_merge_empty_mask_keeps_dummy_media_in_autograd_graph():
    # The empty-media convention: a media-free microbatch still runs the encoder on a dummy
    # segment. The dummy output must stay in the autograd graph with zero gradient signal.
    token_embeddings = torch.randn(2, 4, 3)
    media_token_mask = torch.zeros(2, 4, dtype=torch.bool)
    media_embeddings = torch.randn(1, 3, requires_grad=True)

    merged = merge_media_embeddings(token_embeddings, media_token_mask, media_embeddings)
    merged.sum().backward()

    torch.testing.assert_close(merged, token_embeddings)
    assert media_embeddings.grad is not None
    torch.testing.assert_close(media_embeddings.grad, torch.zeros(1, 3))


@pytest.mark.local
def test_merge_gradients_flow_to_both_sources():
    token_embeddings = torch.randn(1, 4, 3, requires_grad=True)
    media_token_mask = torch.tensor([[False, True, False, True]])
    media_embeddings = torch.randn(2, 3, requires_grad=True)

    merged = merge_media_embeddings(token_embeddings, media_token_mask, media_embeddings)
    merged.sum().backward()

    assert token_embeddings.grad is not None
    assert media_embeddings.grad is not None
    # masked positions receive no gradient in the token embeddings
    torch.testing.assert_close(token_embeddings.grad[0, 1], torch.zeros(3))
    torch.testing.assert_close(token_embeddings.grad[0, 0], torch.ones(3))
    torch.testing.assert_close(media_embeddings.grad, torch.ones(2, 3))


@pytest.mark.local
def test_merge_rejects_count_mismatch():
    token_embeddings = torch.zeros(1, 4, 3)
    media_token_mask = torch.tensor([[False, True, True, False]])
    media_embeddings = torch.zeros(3, 3)

    with pytest.raises(ValueError, match="does not match"):
        merge_media_embeddings(token_embeddings, media_token_mask, media_embeddings)
