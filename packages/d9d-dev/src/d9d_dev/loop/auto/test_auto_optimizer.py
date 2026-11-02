import pytest
from d9d.loop.auto import AutoOptimizerConfig
from pydantic import TypeAdapter, ValidationError


@pytest.mark.local
def test_rejects_unsupported_stochastic_state_dtype():
    config = {"name": "stochastic_adamw", "lr": 1e-3, "state_dtype": "float16"}

    with pytest.raises(ValidationError, match="state_dtype"):
        TypeAdapter(AutoOptimizerConfig).validate_python(config)
