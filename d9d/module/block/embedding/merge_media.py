import torch


def merge_media_embeddings(
    token_embeddings: torch.Tensor,
    media_token_mask: torch.Tensor,
    media_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Replaces placeholder token embeddings with media embeddings.

    Scatters ``media_embeddings`` into the positions of ``token_embeddings`` marked by
    ``media_token_mask``, in row-major order. Gradients flow into both the token embeddings and
    the media embeddings.

    When the mask selects no positions (the empty-media convention: a media-free microbatch ran
    the modality encoder on a dummy segment), the media embeddings are attached to the output
    with a zero-valued contribution instead. This keeps the encoder inside the autograd graph so
    that every rank produces (zero) gradients for its parameters, which is required for gradient
    synchronization collectives to stay aligned across ranks.

    Args:
        token_embeddings: Token embeddings, shape ``(batch, seq, hidden)``.
        media_token_mask: Boolean mask of placeholder positions, shape ``(batch, seq)``.
        media_embeddings: Media token embeddings, shape ``(total_media_tokens, hidden)``.

    Returns:
        The merged embeddings, shape ``(batch, seq, hidden)``.

    Raises:
        ValueError: If the mask selects at least one position and the number of masked positions
            does not match the number of media tokens.
    """
    num_placeholders = int(media_token_mask.sum().item())

    if num_placeholders == 0:
        # Empty-media convention: no placeholders, but the encoder ran on a dummy segment. Attach
        # its output with zero weight to keep the autograd graph structurally identical to the
        # media-carrying case.
        return token_embeddings + (media_embeddings.to(token_embeddings.dtype).sum() * 0.0)

    if num_placeholders != media_embeddings.shape[0]:
        raise ValueError(
            f"Number of media placeholder positions ({num_placeholders}) does not match "
            f"the number of media tokens produced by the encoder ({media_embeddings.shape[0]})."
        )

    return token_embeddings.masked_scatter(media_token_mask.unsqueeze(-1), media_embeddings.to(token_embeddings.dtype))
