import dataclasses

import torch


@dataclasses.dataclass
class SequenceCausalLMHeadShared:
    """The shared input a causal language modeling head consumes.

    Attributes:
        labels: Target token indices for the loss computation. Shape: ``(batch, seq_len)``.
    """

    labels: torch.Tensor


@dataclasses.dataclass
class SequencePoolingHeadShared:
    """The shared input a pooled head (classification or embedding) consumes.

    Attributes:
        pooling_mask: Binary mask of the tokens to pool. You can build it from an attention mask with
            ``d9d.dataset.token_pooling_mask_from_attention_mask``. Shape: ``(batch, seq_len)``.
    """

    pooling_mask: torch.Tensor | None = None


@dataclasses.dataclass
class SequenceCausalLMOutput:
    """The output of a causal language modeling head.

    Attributes:
        logps: Per-token cross-entropy loss (negative log-probabilities). Shape: ``(batch, seq_len)``.
    """

    logps: torch.Tensor


@dataclasses.dataclass
class SequenceClassificationOutput:
    """The output of a classification head.

    Attributes:
        scores: Classification logits. Shape: ``(num_pooled_tokens, num_labels)`` with a pooling mask,
            ``(batch, seq_len, num_labels)`` without one.
    """

    scores: torch.Tensor


@dataclasses.dataclass
class SequenceEmbeddingOutput:
    """The output of an embedding head.

    Attributes:
        embeddings: Pooled embeddings. Shape: ``(num_pooled_tokens, embedding_dim)`` with a pooling mask,
            ``(batch, seq_len, embedding_dim)`` without one.
    """

    embeddings: torch.Tensor
