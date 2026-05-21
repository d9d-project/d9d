import pytest
import torch
from d9d.module.block.positional import (
    LinearRopeScaling,
    NtkRopeScaling,
    RotaryEmbeddingProvider,
    RotaryEmbeddingStyle,
    YarnRopeScaling,
)
from d9d.module.block.positional.rope import prepare_rotary_cos_sin_emb
from torch.testing import assert_close
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

_HEAD_DIM = 64
_ROPE_BASE = 150_000
_MAX_POS = 256
_YARN_FACTOR = 32.0
_YARN_BETA_FAST = 32.0
_YARN_BETA_SLOW = 1.0
_YARN_ORIGINAL_MAX_POS = 4096


def _build_d9d_provider(rope_scaling, max_pos=_MAX_POS) -> RotaryEmbeddingProvider:
    """Helper to construct and initialize the exact d9d RotaryEmbeddingProvider."""
    device = torch.device("cpu")
    dtype = torch.float32

    provider = RotaryEmbeddingProvider(
        rope_base=_ROPE_BASE,
        head_dim=_HEAD_DIM,
        max_position_ids=max_pos,
        style=RotaryEmbeddingStyle.HALF,
        rope_scaling=rope_scaling,
    )
    provider.cos_emb = torch.nn.Buffer(torch.empty(max_pos, _HEAD_DIM, device=device, dtype=dtype), persistent=False)
    provider.sin_emb = torch.nn.Buffer(torch.empty(max_pos, _HEAD_DIM, device=device, dtype=dtype), persistent=False)
    provider.reset_parameters()
    return provider


@pytest.mark.local
def test_provider_with_yarn_matches_hf_gpt_oss_rotary() -> None:
    provider = _build_d9d_provider(
        rope_scaling=YarnRopeScaling(
            factor=_YARN_FACTOR,
            beta_fast=_YARN_BETA_FAST,
            beta_slow=_YARN_BETA_SLOW,
            original_max_position_embeddings=_YARN_ORIGINAL_MAX_POS,
        )
    )

    hf_config = GptOssConfig(
        vocab_size=128,
        hidden_size=_HEAD_DIM * 2,
        intermediate_size=_HEAD_DIM * 4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=_HEAD_DIM,
        max_position_embeddings=_MAX_POS,
        rope_theta=_ROPE_BASE,
        rope_parameters={
            "rope_type": "yarn",
            "factor": _YARN_FACTOR,
            "beta_fast": _YARN_BETA_FAST,
            "beta_slow": _YARN_BETA_SLOW,
            "truncate": False,
            "original_max_position_embeddings": _YARN_ORIGINAL_MAX_POS,
        },
    )
    hf_rotary = GptOssRotaryEmbedding(hf_config)

    position_ids = torch.arange(_MAX_POS, dtype=torch.long).unsqueeze(0)
    d9d_cos, d9d_sin = provider(position_ids)

    dummy_x = torch.empty(1, dtype=torch.float32)
    hf_cos, hf_sin = hf_rotary(dummy_x, position_ids)

    # HF returns (batch, seq_len, head_dim // 2); d9d HALF style returns
    # (batch, seq_len, head_dim) with the second half dimension duplicating the first.
    half = _HEAD_DIM // 2
    assert_close(d9d_cos[..., :half], hf_cos, rtol=1e-5, atol=1e-5)
    assert_close(d9d_cos[..., half:], hf_cos, rtol=1e-5, atol=1e-5)
    assert_close(d9d_sin[..., :half], hf_sin, rtol=1e-5, atol=1e-5)
    assert_close(d9d_sin[..., half:], hf_sin, rtol=1e-5, atol=1e-5)


@pytest.mark.local
def test_provider_with_linear_matches_hf_llama() -> None:
    factor = 4.0
    provider = _build_d9d_provider(rope_scaling=LinearRopeScaling(factor=factor))

    # HF Llama Configuration strictly expects the architecture standard
    hf_config = LlamaConfig(
        hidden_size=_HEAD_DIM * 2,
        intermediate_size=_HEAD_DIM * 4,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=_HEAD_DIM,
        max_position_embeddings=_MAX_POS,
        rope_theta=_ROPE_BASE,
        rope_scaling={"type": "linear", "factor": factor},
    )
    hf_rotary = LlamaRotaryEmbedding(hf_config)

    position_ids = torch.arange(_MAX_POS, dtype=torch.long).unsqueeze(0)
    d9d_cos, d9d_sin = provider(position_ids)

    # Llama requires x dummy shape of [batch, num_heads, seq_len, head_dim]
    dummy_x = torch.empty(1, 1, _MAX_POS, _HEAD_DIM, dtype=torch.float32)
    hf_cos, hf_sin = hf_rotary(dummy_x, position_ids)

    # LlamaRotaryEmbedding returns [batch, 1, seq_len, head_dim] fully repeating channels
    assert_close(d9d_cos, hf_cos.squeeze(1), rtol=1e-5, atol=1e-5)
    assert_close(d9d_sin, hf_sin.squeeze(1), rtol=1e-5, atol=1e-5)


@pytest.mark.local
def test_yarn_rejects_inverted_betas() -> None:
    with pytest.raises(ValueError, match="beta_fast"):
        YarnRopeScaling(
            factor=_YARN_FACTOR,
            beta_fast=1.0,
            beta_slow=32.0,
            original_max_position_embeddings=_YARN_ORIGINAL_MAX_POS,
        )


@pytest.mark.local
def test_provider_without_scaling_is_unchanged() -> None:
    device = torch.device("cpu")
    dtype = torch.float32

    cos_default, sin_default = prepare_rotary_cos_sin_emb(
        rope_base=_ROPE_BASE,
        head_dim=_HEAD_DIM,
        max_position_ids=_MAX_POS,
        device=device,
        dtype=dtype,
        style=RotaryEmbeddingStyle.HALF,
    )

    cos_none, sin_none = prepare_rotary_cos_sin_emb(
        rope_base=_ROPE_BASE,
        head_dim=_HEAD_DIM,
        max_position_ids=_MAX_POS,
        device=device,
        dtype=dtype,
        style=RotaryEmbeddingStyle.HALF,
        rope_scaling=None,
    )

    assert_close(cos_default, cos_none, rtol=0.0, atol=0.0)
    assert_close(sin_default, sin_none, rtol=0.0, atol=0.0)


@pytest.mark.local
def test_provider_with_ntk_matches_hf_llama_with_scaled_base() -> None:
    factor = 4.0
    provider = _build_d9d_provider(rope_scaling=NtkRopeScaling(factor=factor))

    # Static NTK mathematical equivalent in Hugging Face is simply a pre-scaled rope_theta
    scaled_base = _ROPE_BASE * (factor ** (_HEAD_DIM / (_HEAD_DIM - 2)))

    hf_config = LlamaConfig(
        hidden_size=_HEAD_DIM * 2,
        intermediate_size=_HEAD_DIM * 4,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=_HEAD_DIM,
        max_position_embeddings=_MAX_POS,
        rope_theta=scaled_base,
        rope_scaling=None,  # Scaling is intrinsically handled by the manually extended theta
    )
    hf_rotary = LlamaRotaryEmbedding(hf_config)

    position_ids = torch.arange(_MAX_POS, dtype=torch.long).unsqueeze(0)
    d9d_cos, d9d_sin = provider(position_ids)

    # Llama requires dummy shape of [batch, num_heads, seq_len, head_dim]
    dummy_x = torch.empty(1, 1, _MAX_POS, _HEAD_DIM, dtype=torch.float32)
    hf_cos, hf_sin = hf_rotary(dummy_x, position_ids)

    assert_close(d9d_cos, hf_cos.squeeze(1), rtol=1e-5, atol=1e-5)
    assert_close(d9d_sin, hf_sin.squeeze(1), rtol=1e-5, atol=1e-5)
