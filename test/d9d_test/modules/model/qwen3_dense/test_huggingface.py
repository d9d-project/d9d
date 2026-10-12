import pytest
from d9d.module.model.qwen3_dense import (
    Qwen3DenseLayerParameters,
    Qwen3DenseParameters,
    parameters_from_huggingface_qwen3_dense,
)

from d9d_test.modules.model.sequence.catalogue import HF_MODEL_PARAMETERS, QWEN3_DENSE_PARAMETERS, ModelCatalogue

# The config.json of Qwen/Qwen3-0.6B on the Hugging Face Hub, written by Transformers 4.
_TRANSFORMERS_4_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 40960,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

_TRANSFORMERS_4_PARAMETERS = Qwen3DenseParameters(
    layer=Qwen3DenseLayerParameters(
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=16,
        num_key_value_heads=8,
        rms_norm_eps=1e-06,
        head_dim=128,
    ),
    num_hidden_layers=28,
    rope_base=1000000,
    max_position_ids=40960,
    split_vocab_size={"regular": 151936},
    split_vocab_order=["regular"],
)

# The installed Transformers writes the current layout with rope_parameters and layer_types.
_TRANSFORMERS_5_CONFIG = HF_MODEL_PARAMETERS[ModelCatalogue.QWEN3_DENSE].to_dict()


def _changed(config: dict, update: dict, remove: tuple[str, ...] = ()) -> dict:
    changed = {**config, **update}
    for key in remove:
        del changed[key]
    return changed


@pytest.mark.local
def test_transformers_4_config():
    params = parameters_from_huggingface_qwen3_dense(_TRANSFORMERS_4_CONFIG, vocab_split_name="regular")

    assert params == _TRANSFORMERS_4_PARAMETERS


@pytest.mark.local
def test_transformers_5_config():
    # The test covers the layout without rope_theta only while the installed Transformers writes it.
    assert "rope_parameters" in _TRANSFORMERS_5_CONFIG
    assert "rope_theta" not in _TRANSFORMERS_5_CONFIG

    params = parameters_from_huggingface_qwen3_dense(_TRANSFORMERS_5_CONFIG, vocab_split_name="a")

    assert params == QWEN3_DENSE_PARAMETERS


@pytest.mark.local
def test_sliding_window_size_without_sliding_window():
    # Transformers defaults sliding_window to 4096, which has no effect while use_sliding_window is false.
    config = _changed(_TRANSFORMERS_4_CONFIG, {"sliding_window": 4096})

    params = parameters_from_huggingface_qwen3_dense(config, vocab_split_name="regular")

    assert params == _TRANSFORMERS_4_PARAMETERS


@pytest.mark.local
@pytest.mark.parametrize(
    ("config", "match"),
    [
        pytest.param(_changed(_TRANSFORMERS_4_CONFIG, {"model_type": "qwen3_moe"}), "model_type", id="model_type"),
        pytest.param(_changed(_TRANSFORMERS_4_CONFIG, {}, remove=("head_dim",)), "head_dim", id="missing_head_dim"),
        pytest.param(_changed(_TRANSFORMERS_4_CONFIG, {"hidden_act": "gelu"}), "hidden_act", id="hidden_act"),
        pytest.param(_changed(_TRANSFORMERS_4_CONFIG, {"attention_bias": True}), "attention_bias", id="attention_bias"),
        pytest.param(
            _changed(_TRANSFORMERS_4_CONFIG, {"attention_dropout": 0.1}), "attention_dropout", id="attention_dropout"
        ),
        pytest.param(
            _changed(_TRANSFORMERS_4_CONFIG, {"use_sliding_window": True}),
            "use_sliding_window",
            id="use_sliding_window",
        ),
        pytest.param(
            _changed(_TRANSFORMERS_4_CONFIG, {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}),
            "rope_scaling",
            id="rope_scaling",
        ),
        pytest.param(
            _changed(_TRANSFORMERS_5_CONFIG, {"layer_types": ["sliding_attention"] * 8}),
            "layer_types",
            id="layer_types",
        ),
        pytest.param(
            _changed(_TRANSFORMERS_5_CONFIG, {"rope_parameters": {"rope_type": "yarn", "rope_theta": 10000}}),
            "rope_type",
            id="rope_type",
        ),
        pytest.param(
            _changed(
                _TRANSFORMERS_5_CONFIG,
                {"rope_parameters": {"rope_type": "default", "rope_theta": 10000, "factor": 4.0}},
            ),
            "factor",
            id="rope_parameters_factor",
        ),
        pytest.param(
            _changed(_TRANSFORMERS_4_CONFIG, {}, remove=("rope_theta",)), "exactly one of rope_theta", id="no_rope"
        ),
        pytest.param(
            _changed(_TRANSFORMERS_4_CONFIG, {"rope_parameters": {"rope_type": "default", "rope_theta": 1000000}}),
            "exactly one of rope_theta",
            id="both_rope_layouts",
        ),
    ],
)
def test_unsupported_config(config: dict, match: str):
    with pytest.raises(ValueError, match=match):
        parameters_from_huggingface_qwen3_dense(config, vocab_split_name="regular")
