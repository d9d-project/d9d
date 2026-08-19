"""This package provides utilities and torch.utils.data.Dataset implementations."""

from .batch_iterator import FixedCountMicrobatchPacker, num_microbatches_for_global_batch
from .buffer_sorted import BufferSortedDataset, DatasetImplementingSortKeyProtocol
from .padding import PaddingSide1D, pad_stack_1d
from .pooling import TokenPoolingType, token_pooling_mask_from_attention_mask
from .sequence_packing import (
    DatasetImplementingSampleLengthProtocol,
    PackedSequence,
    SequencePackingDataset,
    StreamingSequencePackingDataset,
    pack_collate,
    pack_samples,
)
from .sharded import ShardedDataset, ShardIndexingMode, shard_dataset_data_parallel

__all__ = [
    "BufferSortedDataset",
    "DatasetImplementingSampleLengthProtocol",
    "DatasetImplementingSortKeyProtocol",
    "FixedCountMicrobatchPacker",
    "PackedSequence",
    "PaddingSide1D",
    "SequencePackingDataset",
    "ShardIndexingMode",
    "ShardedDataset",
    "StreamingSequencePackingDataset",
    "TokenPoolingType",
    "num_microbatches_for_global_batch",
    "pack_collate",
    "pack_samples",
    "pad_stack_1d",
    "shard_dataset_data_parallel",
    "token_pooling_mask_from_attention_mask",
]
