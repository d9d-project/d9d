import pytest
import torch
from d9d.module.block.positional import (
    MultimodalRotaryEmbeddingProvider,
    RotaryEmbeddingProvider,
    RotaryEmbeddingStyle,
)
from torch.testing import assert_close
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

_ROPE_DIM = 64
_ROPE_BASE = 5_000_000
_MAX_POS = 512
_MROPE_SECTION = (24, 4, 4)
_BATCH = 2
_SEQ = 128


def _build_d9d_provider(interleaved: bool) -> MultimodalRotaryEmbeddingProvider:
    provider = MultimodalRotaryEmbeddingProvider(
        rope_base=_ROPE_BASE,
        rope_dim=_ROPE_DIM,
        max_position_ids=_MAX_POS,
        mrope_section=_MROPE_SECTION,
        interleaved=interleaved,
    )
    provider.reset_parameters()
    return provider


def _build_3d_position_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(42)

    # text prefix: all three planes equal; then a media block: planes diverge
    position_ids = torch.zeros(3, _BATCH, _SEQ, dtype=torch.long)
    for batch_idx in range(_BATCH):
        text_len = 40 + batch_idx * 10
        position_ids[:, batch_idx, :text_len] = torch.arange(text_len)
        media = torch.randint(0, _MAX_POS, (3, _SEQ - text_len), generator=generator)
        position_ids[:, batch_idx, text_len:] = media
    return position_ids


@pytest.mark.local
def test_mrope_matches_hf_qwen3_vl_interleaved():
    provider = _build_d9d_provider(interleaved=True)

    hf_rotary = Qwen3VLTextRotaryEmbedding(
        Qwen3VLTextConfig(
            hidden_size=1024,
            num_attention_heads=16,
            head_dim=_ROPE_DIM,
            rope_theta=_ROPE_BASE,
            max_position_embeddings=_MAX_POS,
            rope_scaling={"rope_type": "default", "mrope_section": list(_MROPE_SECTION), "mrope_interleaved": True},
        )
    )

    position_ids = _build_3d_position_ids()

    ref_dtype_tensor = torch.zeros(1, dtype=torch.float32)
    cos_hf, sin_hf = hf_rotary(ref_dtype_tensor, position_ids)

    cos_d9d, sin_d9d = provider(position_ids)

    assert_close(cos_d9d, cos_hf, atol=1e-5, rtol=1e-5)
    assert_close(sin_d9d, sin_hf, atol=1e-5, rtol=1e-5)


@pytest.mark.local
def test_mrope_equals_rope_for_text_only_positions():
    mrope_provider = _build_d9d_provider(interleaved=True)

    rope_provider = RotaryEmbeddingProvider(
        rope_base=_ROPE_BASE,
        head_dim=_ROPE_DIM,
        max_position_ids=_MAX_POS,
        style=RotaryEmbeddingStyle.HALF,
    )
    rope_provider.reset_parameters()

    text_positions = torch.arange(_SEQ).view(1, -1).expand(_BATCH, -1)
    position_ids_3d = text_positions.unsqueeze(0).expand(3, -1, -1)

    cos_mrope, sin_mrope = mrope_provider(position_ids_3d)
    cos_rope, sin_rope = rope_provider(text_positions)

    assert_close(cos_mrope, cos_rope)
    assert_close(sin_mrope, sin_rope)


@pytest.mark.local
def test_mrope_rejects_invalid_section():
    with pytest.raises(ValueError, match="mrope_section"):
        MultimodalRotaryEmbeddingProvider(
            rope_base=_ROPE_BASE,
            rope_dim=_ROPE_DIM,
            max_position_ids=_MAX_POS,
            mrope_section=(10, 10, 10),
        )


@pytest.mark.local
def test_mrope_rejects_2d_position_ids():
    provider = _build_d9d_provider(interleaved=True)

    with pytest.raises(ValueError, match="position_ids"):
        provider(torch.zeros(_BATCH, _SEQ, dtype=torch.long))
