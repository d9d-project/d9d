import pytest
from d9d.loop.config import InferenceConfig, ParameterRuntimeConfig
from pydantic import ValidationError


@pytest.mark.local
@pytest.mark.parametrize(
    ("section", "error"),
    [
        pytest.param({"compute_dtype": "bfloat16", "memory_budget_mib": 16, "bucket_size_mib": 8}, None, id="valid"),
        pytest.param({"compute_dtype": "int64"}, "compute_dtype", id="integer_compute_dtype"),
        pytest.param({"memory_budget_mib": -1}, "memory_budget_mib", id="negative_budget"),
        pytest.param({"bucket_size_mib": 0}, "bucket_size_mib", id="empty_bucket"),
    ],
)
def test_inference_config_validates_the_runtime_section(section, error):
    raw = {
        "determinism": {"base_seed": 42},
        "gc": {"period_steps": 100},
        "checkpointing": {"save_dir": "checkpoints", "period_steps": 100},
        "parameter_runtime": section,
    }
    if error is not None:
        with pytest.raises(ValidationError, match=error):
            InferenceConfig.model_validate(raw)
        return

    assert InferenceConfig.model_validate(raw).parameter_runtime == ParameterRuntimeConfig(**section)
