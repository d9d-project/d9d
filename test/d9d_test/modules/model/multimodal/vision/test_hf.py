import pytest
import torch
from d9d.module.base import MediaSegments
from d9d.module.block.attention.sdpa import TorchSdpaBackendConfig
from d9d.module.model.qwen3_5 import (
    Qwen3p5VisionModel,
    Qwen3p5VisionParameters,
    mapper_from_huggingface_qwen3p5_vision,
)
from d9d.module.model.qwen3_5_moe import (
    Qwen3p5MoEVisionModel,
    Qwen3p5MoEVisionParameters,
    mapper_from_huggingface_qwen3p5_moe_vision,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights, torch_seed
from d9d_test.modules.model.multimodal.vision.batch import (
    VISION_HIDDEN_SIZE,
    VISION_IN_CHANNELS,
    VISION_INTERMEDIATE_SIZE,
    VISION_NORM_EPS,
    VISION_NUM_HEADS,
    VISION_PATCH_SIZE,
    VISION_SPATIAL_MERGE_SIZE,
    VISION_TEMPORAL_PATCH_SIZE,
    build_vision_inputs,
    materialize_vision_inputs,
)

_DEPTH = 2
_OUT_HIDDEN_SIZE = 128
_NUM_POSITION_EMBEDDINGS = 64  # 8x8 learned grid
_MAX_GRID_SIDE = 64

_D9D_VISION_PARAMS = {
    "depth": _DEPTH,
    "hidden_size": VISION_HIDDEN_SIZE,
    "intermediate_size": VISION_INTERMEDIATE_SIZE,
    "num_attention_heads": VISION_NUM_HEADS,
    "in_channels": VISION_IN_CHANNELS,
    "patch_size": VISION_PATCH_SIZE,
    "temporal_patch_size": VISION_TEMPORAL_PATCH_SIZE,
    "spatial_merge_size": VISION_SPATIAL_MERGE_SIZE,
    "out_hidden_size": _OUT_HIDDEN_SIZE,
    "num_position_embeddings": _NUM_POSITION_EMBEDDINGS,
    "max_grid_side": _MAX_GRID_SIDE,
    "norm_eps": VISION_NORM_EPS,
}

_HF_VISION_PARAMS = {
    "depth": _DEPTH,
    "hidden_size": VISION_HIDDEN_SIZE,
    "intermediate_size": VISION_INTERMEDIATE_SIZE,
    "num_heads": VISION_NUM_HEADS,
    "in_channels": VISION_IN_CHANNELS,
    "patch_size": VISION_PATCH_SIZE,
    "spatial_merge_size": VISION_SPATIAL_MERGE_SIZE,
    "temporal_patch_size": VISION_TEMPORAL_PATCH_SIZE,
    "out_hidden_size": _OUT_HIDDEN_SIZE,
    "num_position_embeddings": _NUM_POSITION_EMBEDDINGS,
    "_attn_implementation": "sdpa",
}

_VARIANTS = {
    "qwen3_5": (
        lambda: Qwen3_5VisionModel(Qwen3_5VisionConfig(**_HF_VISION_PARAMS)),
        lambda: Qwen3p5VisionModel(
            Qwen3p5VisionParameters(**_D9D_VISION_PARAMS), sdpa_backend=TorchSdpaBackendConfig()
        ),
        lambda: mapper_from_huggingface_qwen3p5_vision(Qwen3p5VisionParameters(**_D9D_VISION_PARAMS)),
    ),
    "qwen3_5_moe": (
        lambda: Qwen3_5MoeVisionModel(Qwen3_5MoeVisionConfig(**_HF_VISION_PARAMS)),
        lambda: Qwen3p5MoEVisionModel(
            Qwen3p5MoEVisionParameters(**_D9D_VISION_PARAMS), sdpa_backend=TorchSdpaBackendConfig()
        ),
        lambda: mapper_from_huggingface_qwen3p5_moe_vision(Qwen3p5MoEVisionParameters(**_D9D_VISION_PARAMS)),
    ),
}


@pytest.mark.local
@pytest.mark.parametrize("variant", list(_VARIANTS))
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(variant: str, dtype: torch.dtype):
    hf_factory, d9d_factory, mapper_factory = _VARIANTS[variant]
    init = build_vision_inputs(dtype)
    mapper = mapper_factory()

    # HF
    inputs_hf = materialize_vision_inputs(init)
    with torch_seed(42):
        module_hf = hf_factory().cuda().to(dtype)

    outputs_hf = module_hf(inputs_hf.features + inputs_hf.pre, grid_thw=inputs_hf.grid_thw)
    merged_hf = outputs_hf.pooler_output
    merged_hf.float().mean().backward()

    # d9d
    inputs_d9d = materialize_vision_inputs(init)
    with torch_seed(43):
        module_d9d = d9d_factory().cuda().to(dtype)
        module_d9d.reset_parameters()
    clone_module_weights(from_module=module_hf, to_module=module_d9d, map_with=mapper)

    merged_d9d = module_d9d(MediaSegments(features=inputs_d9d.features + inputs_d9d.pre, grid_thw=inputs_d9d.grid_thw))
    merged_d9d.float().mean().backward()

    # Check
    torch.testing.assert_close(merged_d9d, merged_hf, atol=3e-2, rtol=1e-2)
    torch.testing.assert_close(inputs_d9d.pre.grad, inputs_hf.pre.grad, atol=1e-5, rtol=0.01)
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)
