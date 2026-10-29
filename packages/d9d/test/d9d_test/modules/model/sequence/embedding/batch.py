from dataclasses import dataclass

import torch
from d9d.core.dist_context import DistributedContext
from d9d.dataset import TokenPoolingType, token_pooling_mask_from_attention_mask

from d9d_test.modules.helper.distributed import shard_batch_dim
from d9d_test.modules.model.sequence.batch import SequenceBatch, build_sequence_batch, shard_sequence_batch


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
