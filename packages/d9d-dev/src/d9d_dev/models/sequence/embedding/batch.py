from dataclasses import dataclass

import torch
from d9d.data import TokenPoolingType, token_pooling_mask_from_attention_mask
from d9d.dist_context import DistributedContext

from d9d_dev.models.sequence.batch import SequenceBatch, build_sequence_batch, shard_sequence_batch
from d9d_dev.nn.helper.distributed import shard_batch_dim


@dataclass
class EmbeddingBatch:
    sequence: SequenceBatch
    pooling_mask: torch.Tensor


def build_embedding_batch(
    device: torch.device | str = "cuda",
) -> EmbeddingBatch:
    batch = build_sequence_batch(device=device)

    return EmbeddingBatch(
        sequence=batch,
        pooling_mask=token_pooling_mask_from_attention_mask(batch.attention_mask, TokenPoolingType.last),
    )


def shard_embedding_batch(batch: EmbeddingBatch, dist_ctx: DistributedContext) -> EmbeddingBatch:
    return EmbeddingBatch(
        sequence=shard_sequence_batch(batch.sequence, dist_ctx),
        pooling_mask=shard_batch_dim(batch.pooling_mask, dist_ctx),
    )
