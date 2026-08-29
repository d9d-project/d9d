from .head import parallelize_causal_lm_head, parallelize_classification_head, parallelize_embedding_head
from .qwen3_5 import parallelize_qwen3p5_model, parallelize_qwen3p5_vision
from .qwen3_5_moe import parallelize_qwen3p5_moe_model, parallelize_qwen3p5_moe_vision
from .qwen3_dense import parallelize_qwen3_dense_model
from .qwen3_moe import parallelize_qwen3_moe_model

__all__ = [
    "parallelize_causal_lm_head",
    "parallelize_classification_head",
    "parallelize_embedding_head",
    "parallelize_qwen3_dense_model",
    "parallelize_qwen3_moe_model",
    "parallelize_qwen3p5_model",
    "parallelize_qwen3p5_moe_model",
    "parallelize_qwen3p5_moe_vision",
    "parallelize_qwen3p5_vision",
]
