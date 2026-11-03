import pytest
import torch
from d9d.loop.config import GradientManagerConfig, InferenceConfig, ParameterRuntimeConfig
from d9d.loop.config.types import dtype_from_name
from pydantic import ValidationError


@pytest.mark.local
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_compute_dtype_is_validated_and_resolved(dtype):
    config = ParameterRuntimeConfig(compute_dtype=dtype)

    assert dtype_from_name(config.compute_dtype) is getattr(torch, dtype)


@pytest.mark.local
@pytest.mark.parametrize("dtype", ["float", "int64", "Tensor", "not_a_dtype"])
def test_rejects_invalid_compute_dtype(dtype):
    with pytest.raises(ValidationError, match="compute_dtype"):
        ParameterRuntimeConfig.model_validate({"compute_dtype": dtype})


@pytest.mark.local
@pytest.mark.parametrize("key", ["memory_budget_mib", "bucket_size_mb"])
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_nonpositive_sizes(key, value):
    with pytest.raises(ValidationError, match=key):
        ParameterRuntimeConfig.model_validate({key: value})


@pytest.mark.local
def test_inference_config_has_runtime_options():
    config = InferenceConfig.model_validate(
        {
            "determinism": {"base_seed": 42},
            "gc": {"period_steps": 100},
            "checkpointing": {"save_dir": "checkpoints", "period_steps": 100},
            "parameter_runtime": {"compute_dtype": "bfloat16", "memory_budget_mib": 16},
        }
    )

    assert config.parameter_runtime.compute_dtype == "bfloat16"
    assert config.parameter_runtime.memory_budget_mib == 16


@pytest.mark.local
def test_rejects_invalid_gradient_dtype():
    with pytest.raises(ValidationError):
        GradientManagerConfig.model_validate({"grad_dtype": "int64"})
