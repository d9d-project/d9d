from collections.abc import Mapping
from dataclasses import dataclass

import torch
from d9d.core.dist_context import DistributedContext

from d9d_test.modules.helper.distributed import shard_batch_dim

_VOCAB_SIZE = 90
_BATCH_SIZE = 16
_SEQ_LEN = 33
_PAD_TOKEN_ID = 99
_PAD_LENGTH = {1: 1, 2: 5}


@dataclass
class SequenceBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


def _apply_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
    pad_lengths: Mapping[int, int],
) -> None:
    for row, n_pad in pad_lengths.items():
        if n_pad <= 0:
            continue
        input_ids[row, -n_pad:] = int(pad_token_id)
        attention_mask[row, -n_pad:] = 0


def build_sequence_batch(
    device: torch.device | str = "cuda",
) -> SequenceBatch:
    input_ids = torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, _SEQ_LEN), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    _apply_padding(
        input_ids,
        attention_mask,
        pad_token_id=_PAD_TOKEN_ID,
        pad_lengths=_PAD_LENGTH,
    )

    position_ids = torch.arange(_SEQ_LEN, device=device, dtype=torch.long).unsqueeze(0).repeat(_BATCH_SIZE, 1)

    return SequenceBatch(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)


def shard_sequence_batch(batch: SequenceBatch, dist_ctx: DistributedContext) -> SequenceBatch:
    return SequenceBatch(
        input_ids=shard_batch_dim(batch.input_ids, dist_ctx),
        attention_mask=shard_batch_dim(batch.attention_mask, dist_ctx),
        position_ids=shard_batch_dim(batch.position_ids, dist_ctx),
    )
