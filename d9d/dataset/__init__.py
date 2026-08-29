"""This package provides utilities and torch.utils.data.Dataset implementations."""

from .batch_iterator import FixedCountMicrobatchPacker, num_microbatches_for_global_batch
from .buffer_sorted import BufferSortedDataset, DatasetImplementingSortKeyProtocol
from .multimodal import compute_multimodal_position_ids, pad_empty_media
from .padding import PaddingSide1D, pad_stack_1d
from .pooling import TokenPoolingType, token_pooling_mask_from_attention_mask
from .sharded import ShardedDataset, ShardIndexingMode, shard_dataset_data_parallel

__all__ = [
    "BufferSortedDataset",
    "DatasetImplementingSortKeyProtocol",
    "FixedCountMicrobatchPacker",
    "PaddingSide1D",
    "ShardIndexingMode",
    "ShardedDataset",
    "TokenPoolingType",
    "compute_multimodal_position_ids",
    "num_microbatches_for_global_batch",
    "pad_empty_media",
    "pad_stack_1d",
    "shard_dataset_data_parallel",
    "token_pooling_mask_from_attention_mask",
]
