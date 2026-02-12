"""
This package provides utilities and torch.utils.data.Dataset implementations.
"""

from .buffer_sorted import BufferSortedDataset, DatasetImplementingSortKeyProtocol
from .padding import PaddingSide1D, pad_stack_1d
from .pooling import TokenPoolingType, token_pooling_mask_from_attention_mask
from .sharded import ShardedDataset, ShardIndexingMode, shard_dataset_data_parallel

__all__ = [
    "BufferSortedDataset",
    "DatasetImplementingSortKeyProtocol",
    "PaddingSide1D",
    "ShardIndexingMode",
    "ShardedDataset",
    "TokenPoolingType",
    "pad_stack_1d",
    "shard_dataset_data_parallel",
    "token_pooling_mask_from_attention_mask"
]
