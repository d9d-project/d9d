import dataclasses

import torch


@dataclasses.dataclass
class SequenceCausalLMHeadShared:
    """The shared input a causal language modeling head consumes.

    Attributes:
        labels: Target tokens for the loss computation.
    """

    labels: torch.Tensor


@dataclasses.dataclass
class SequencePoolingHeadShared:
    """The shared input a pooled head (classification/embedding) consumes.

    Attributes:
        pooling_mask: Binary mask indicating which token(s) to pool. You can use
            ``d9d.dataset.token_pooling_mask_from_attention_mask`` to build it from an attention mask.
    """

    pooling_mask: torch.Tensor | None = None


@dataclasses.dataclass
class SequenceCausalLMOutput:
    """The output of a causal language modeling head.

    Attributes:
        logps: Per-token log-probabilities / loss, shape ``[batch, seq]``.
    """

    logps: torch.Tensor


@dataclasses.dataclass
class SequenceClassificationOutput:
    """The output of a classification head.

    Attributes:
        scores: Classification logits, shape ``[num_pooled_tokens, num_labels]`` when a pooling mask
            selects tokens, or ``[batch, seq, num_labels]`` when no pooling mask is used.
    """

    scores: torch.Tensor


@dataclasses.dataclass
class SequenceEmbeddingOutput:
    """The output of an embedding head.

    Attributes:
        embeddings: Pooled embeddings, shape ``[num_pooled_tokens, embedding_dim]`` when a pooling mask
            selects tokens, or ``[batch, seq, embedding_dim]`` when no pooling mask is used.
    """

    embeddings: torch.Tensor
