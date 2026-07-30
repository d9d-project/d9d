import pytest
import torch
from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperPrefixScope
from d9d.model_state.mapper.leaf import ModelStateMapperIdentity, ModelStateMapperRename
from d9d.module.base import MediaSegments, ModuleLateInit
from d9d.module.block.attention.sdpa import TorchSdpaBackendConfig
from d9d.module.block.vision import (
    InterpolatedPositionEmbedding,
    PatchEmbedding,
    SpatialPatchMerger,
    VisionBlock,
    VisionRotaryEmbedding2D,
    segment_cu_seqlens,
)
from torch import nn
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

from d9d_test.modules.block.vision.batch import (
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
from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights, torch_seed

_DEPTH = 2
_OUT_HIDDEN_SIZE = 128
_NUM_POSITION_EMBEDDINGS = 64  # 8x8 learned grid
_MAX_GRID_SIDE = 64


class _VisionTower(nn.Module, ModuleLateInit):
    """A minimal vision tower assembled from d9d vision blocks (Qwen3.5 layout)."""

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            hidden_size=VISION_HIDDEN_SIZE,
            in_channels=VISION_IN_CHANNELS,
            patch_size=VISION_PATCH_SIZE,
            temporal_patch_size=VISION_TEMPORAL_PATCH_SIZE,
        )
        self.pos_embed = InterpolatedPositionEmbedding(
            hidden_size=VISION_HIDDEN_SIZE,
            num_position_embeddings=_NUM_POSITION_EMBEDDINGS,
            spatial_merge_size=VISION_SPATIAL_MERGE_SIZE,
        )
        self.rotary = VisionRotaryEmbedding2D(
            head_dim=VISION_HIDDEN_SIZE // VISION_NUM_HEADS,
            max_grid_side=_MAX_GRID_SIDE,
            spatial_merge_size=VISION_SPATIAL_MERGE_SIZE,
        )
        self.blocks = nn.ModuleList(
            [
                VisionBlock(
                    hidden_size=VISION_HIDDEN_SIZE,
                    intermediate_size=VISION_INTERMEDIATE_SIZE,
                    num_attention_heads=VISION_NUM_HEADS,
                    norm_eps=VISION_NORM_EPS,
                    sdpa_backend=TorchSdpaBackendConfig(),
                )
                for _ in range(_DEPTH)
            ]
        )
        self.merger = SpatialPatchMerger(
            hidden_size=VISION_HIDDEN_SIZE,
            out_hidden_size=_OUT_HIDDEN_SIZE,
            spatial_merge_size=VISION_SPATIAL_MERGE_SIZE,
            norm_eps=VISION_NORM_EPS,
        )

    def forward(self, media: MediaSegments) -> torch.Tensor:
        hidden_states = self.patch_embed(media.features)
        hidden_states = hidden_states + self.pos_embed(media.grid_thw)

        position_embeddings = self.rotary(media.grid_thw)
        cu_seqlens, max_seqlen = segment_cu_seqlens(media.grid_thw)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
            )

        return self.merger(hidden_states)

    def reset_parameters(self):
        self.patch_embed.reset_parameters()
        self.pos_embed.reset_parameters()
        self.rotary.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.merger.reset_parameters()


def _mapper_from_hf_to_d9d() -> ModelStateMapper:
    def _block_mapper() -> ModelStateMapper:
        return ModelStateMapperParallel(
            [
                *(
                    ModelStateMapperIdentity(f"{name}.{param}")
                    for name in ("norm1", "norm2", "attn.qkv", "attn.proj")
                    for param in ("weight", "bias")
                ),
                *(
                    ModelStateMapperRename(f"mlp.linear_fc{i}.{param}", f"mlp.fc{i}.{param}")
                    for i in (1, 2)
                    for param in ("weight", "bias")
                ),
            ]
        )

    return ModelStateMapperParallel(
        [
            ModelStateMapperIdentity("patch_embed.proj.weight"),
            ModelStateMapperIdentity("patch_embed.proj.bias"),
            ModelStateMapperRename("pos_embed.weight", "pos_embed.pos_embed.weight"),
            *(
                ModelStateMapperPrefixScope(_block_mapper(), source_prefix=f"blocks.{i}.", target_prefix=f"blocks.{i}.")
                for i in range(_DEPTH)
            ),
            ModelStateMapperIdentity("merger.norm.weight"),
            ModelStateMapperIdentity("merger.norm.bias"),
            *(
                ModelStateMapperRename(f"merger.linear_fc{i}.{param}", f"merger.fc{i}.{param}")
                for i in (1, 2)
                for param in ("weight", "bias")
            ),
        ]
    )


def build_hf(dtype: torch.dtype) -> Qwen3_5MoeVisionModel:
    with torch_seed(42):
        return (
            Qwen3_5MoeVisionModel(
                Qwen3_5MoeVisionConfig(
                    depth=_DEPTH,
                    hidden_size=VISION_HIDDEN_SIZE,
                    intermediate_size=VISION_INTERMEDIATE_SIZE,
                    num_heads=VISION_NUM_HEADS,
                    in_channels=VISION_IN_CHANNELS,
                    patch_size=VISION_PATCH_SIZE,
                    spatial_merge_size=VISION_SPATIAL_MERGE_SIZE,
                    temporal_patch_size=VISION_TEMPORAL_PATCH_SIZE,
                    out_hidden_size=_OUT_HIDDEN_SIZE,
                    num_position_embeddings=_NUM_POSITION_EMBEDDINGS,
                    _attn_implementation="sdpa",
                )
            )
            .cuda()
            .to(dtype)
        )


def build_d9d(dtype: torch.dtype) -> _VisionTower:
    with torch_seed(43):
        tower = _VisionTower().cuda().to(dtype)
        tower.reset_parameters()
    return tower


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(dtype: torch.dtype):
    init = build_vision_inputs(dtype)
    mapper = _mapper_from_hf_to_d9d()

    # HF
    inputs_hf = materialize_vision_inputs(init)
    module_hf = build_hf(dtype)

    outputs_hf = module_hf(inputs_hf.features + inputs_hf.pre, grid_thw=inputs_hf.grid_thw)
    merged_hf = outputs_hf.pooler_output
    merged_hf.float().mean().backward()

    # d9d
    inputs_d9d = materialize_vision_inputs(init)
    module_d9d = build_d9d(dtype)
    clone_module_weights(from_module=module_hf, to_module=module_d9d, map_with=mapper)

    merged_d9d = module_d9d(MediaSegments(features=inputs_d9d.features + inputs_d9d.pre, grid_thw=inputs_d9d.grid_thw))
    merged_d9d.float().mean().backward()

    # Check
    torch.testing.assert_close(merged_d9d, merged_hf, atol=3e-2, rtol=1e-2)
    torch.testing.assert_close(inputs_d9d.pre.grad, inputs_hf.pre.grad, atol=1e-5, rtol=0.01)
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)
