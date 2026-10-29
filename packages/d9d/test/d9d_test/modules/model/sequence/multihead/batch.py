from dataclasses import dataclass

import torch
from d9d.core.dist_context import DistributedContext
from d9d.dataset import TokenPoolingType, token_pooling_mask_from_attention_mask

from d9d_test.modules.helper.distributed import shard_batch_dim
from d9d_test.modules.model.sequence.batch import SequenceBatch, build_sequence_batch, shard_sequence_batch

_IGNORE_INDEX = -100


@dataclass
class MultiHeadBatch:
    sequence: SequenceBatch

    lm_labels: torch.Tensor
    pooling_mask: torch.Tensor
    cls_labels: torch.Tensor


def build_multihead_batch(
    num_labels: int,
    device: torch.device | str = "cuda",
) -> MultiHeadBatch:
    # One shared sequence feeds both heads: per-token labels for the LM head and a
    # last-token pooling mask + per-sequence labels for the classification head.
    batch = build_sequence_batch(device=device)

    lm_labels = batch.input_ids.clone()
    lm_labels[batch.attention_mask == 0] = _IGNORE_INDEX

    pooling_mask = token_pooling_mask_from_attention_mask(batch.attention_mask, TokenPoolingType.last)
    cls_labels = torch.randint(0, int(num_labels), (batch.input_ids.shape[0],), device=device, dtype=torch.long)

    return MultiHeadBatch(
        sequence=batch,
        lm_labels=lm_labels,
        pooling_mask=pooling_mask,
        cls_labels=cls_labels,
    )


def shard_multihead_batch(batch: MultiHeadBatch, dist_ctx: DistributedContext) -> MultiHeadBatch:
    return MultiHeadBatch(
        sequence=shard_sequence_batch(batch.sequence, dist_ctx),
        lm_labels=shard_batch_dim(batch.lm_labels, dist_ctx),
        pooling_mask=shard_batch_dim(batch.pooling_mask, dist_ctx),
        cls_labels=shard_batch_dim(batch.cls_labels, dist_ctx),
    )
