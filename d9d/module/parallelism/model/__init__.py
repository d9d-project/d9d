from .head import parallelize_task_head
from .qwen3_dense import parallelize_qwen3_dense_model
from .qwen3_moe import parallelize_qwen3_moe_model

__all__ = [
    "parallelize_qwen3_dense_model",
    "parallelize_qwen3_moe_model",
    "parallelize_task_head",
]
