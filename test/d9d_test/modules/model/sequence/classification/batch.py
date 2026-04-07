from dataclasses import dataclass

import torch
from d9d.core.dist_context import DistributedContext
from d9d.dataset import TokenPoolingType, token_pooling_mask_from_attention_mask

from d9d_test.modules.helper.distributed import shard_batch_dim
from d9d_test.modules.model.sequence.batch import SequenceBatch, build_sequence_batch, shard_sequence_batch
from d9d_test.modules.model.sequence.classification.catalogue import NUM_LABELS_CLS


@dataclass
class ClassificationBatch:
    sequence: SequenceBatch

    labels: torch.Tensor
    pooling_mask: torch.Tensor


def build_classification_batch(
    device: torch.device | str = "cuda",
) -> ClassificationBatch:
    batch = build_sequence_batch(
        device=device,
    )

    labels = torch.randint(0, int(NUM_LABELS_CLS), (batch.position_ids.shape[0],), device=device, dtype=torch.long)

    return ClassificationBatch(
        sequence=batch,
        labels=labels,
        pooling_mask=token_pooling_mask_from_attention_mask(batch.attention_mask, TokenPoolingType.last),
    )


def shard_classification_batch(batch: ClassificationBatch, dist_ctx: DistributedContext) -> ClassificationBatch:
    return ClassificationBatch(
        sequence=shard_sequence_batch(batch.sequence, dist_ctx),
        labels=shard_batch_dim(batch.labels, dist_ctx),
        pooling_mask=shard_batch_dim(batch.pooling_mask, dist_ctx),
    )
