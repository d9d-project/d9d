from collections.abc import Mapping
from dataclasses import dataclass

import torch
from d9d.core.dist_context import DistributedContext

from d9d_test.modules.helper.distributed import shard_batch_dim
from d9d_test.modules.model.sequence.batch import SequenceBatch, build_sequence_batch, shard_sequence_batch

_IGNORE_INDEX = -100
_IGNORE_PREFIX = {0: 119}


@dataclass
class CausalLMBatch:
    sequence: SequenceBatch

    labels: torch.Tensor


def _apply_ignore_prefix(
    labels_causal_lm: torch.Tensor,
    ignore_index: int,
    ignore_prefix: Mapping[int, int],
) -> None:
    for row, prefix_len in ignore_prefix.items():
        if prefix_len <= 0:
            continue
        labels_causal_lm[row, :prefix_len] = ignore_index


def build_causal_lm_batch(
    device: torch.device | str = "cuda",
) -> CausalLMBatch:
    batch = build_sequence_batch(
        device=device,
    )

    labels = batch.input_ids.clone()
    labels[batch.attention_mask == 0] = _IGNORE_INDEX
    _apply_ignore_prefix(
        labels,
        ignore_index=_IGNORE_INDEX,
        ignore_prefix=_IGNORE_PREFIX,
    )

    return CausalLMBatch(
        sequence=batch,
        labels=labels,
    )


def shard_causal_lm_batch(batch: CausalLMBatch, dist_ctx: DistributedContext) -> CausalLMBatch:
    return CausalLMBatch(
        sequence=shard_sequence_batch(batch.sequence, dist_ctx),
        labels=shard_batch_dim(batch.labels, dist_ctx),
    )
