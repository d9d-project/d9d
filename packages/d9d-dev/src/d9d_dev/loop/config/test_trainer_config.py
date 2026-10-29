import pytest
from d9d.loop.config import TrainerConfig
from d9d.pipelining.factory import PipelineScheduleGPipeConfig
from pydantic import ValidationError

_REQUIRED = {
    "run": {"name": "test"},
    "logging": {"period_steps": 10, "tracker": {"provider": "null"}},
    "determinism": {"base_seed": 42},
    "gc": {"period_steps": 100},
    "checkpointing": {"save_dir": "checkpoints", "period_steps": 100},
    "gradient_clipping": {"max_norm": 1.0},
}


@pytest.mark.local
def test_fills_defaults_for_optional_sections():
    config = TrainerConfig.model_validate(_REQUIRED)

    assert config.schedule.total_steps is None
    assert config.data_prefetch.prefetch_factor == 1
    assert config.pipelining.schedule == PipelineScheduleGPipeConfig()
    assert config.model_stage_factory.source_checkpoint is None
    assert not config.model_stage_factory.checkpoint_only_trainable_parameters
    assert config.checkpointing.num_to_keep is None
    assert config.profiling is None
    assert config.gradient_manager.grad_dtype is None
    assert config.gradient_manager.bucket_size_mb == 32


@pytest.mark.local
@pytest.mark.parametrize(
    "config",
    [
        {**_REQUIRED, "chekpointing": {}},
        {**_REQUIRED, "gradient_clipping": {"max_norm": 1.0, "log_total_steps": 10}},
    ],
)
def test_rejects_unknown_keys(config):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainerConfig.model_validate(config)
