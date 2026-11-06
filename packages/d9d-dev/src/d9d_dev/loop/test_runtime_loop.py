import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from d9d.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.loop.auto import (
    AutoDataConfig,
    AutoDataProvider,
    AutoLRSchedulerConfig,
    AutoLRSchedulerProvider,
    AutoOptimizerConfig,
    AutoOptimizerProvider,
)
from d9d.loop.config import TrainerConfig
from d9d.loop.control import (
    BuildForwardInputsContext,
    BuildForwardInputsResult,
    ComputeLossContext,
    ComputeLossResult,
    InitializeModelStageContext,
    InitializeModelStageResult,
    ModelProvider,
    ParallelizeModelStageContext,
    PrepareExportModelStageContext,
    PrepareExportModelStageResult,
    RegisterModelEventsContext,
    TrainTask,
)
from d9d.loop.event.catalogue import EVENT_TRAIN_STEP_POST, EventStepContext
from d9d.loop.run import SleepTag, Trainer, TrainingConfigurator
from d9d.model_state.mapper.adapters import identity_mapper_from_module
from d9d.parallel import parallelize_hsdp
from d9d.pipelining.api import ModuleSupportsPipelining, PipelineStageInfo, StageBoundary, TensorSpec
from pydantic import TypeAdapter
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import Dataset

_NUM_FEATURES = 16
_HIDDEN_SIZE = 64
_TOTAL_STEPS = 6
_HALF_STEPS = _TOTAL_STEPS // 2


@dataclasses.dataclass
class _Sample:
    x: torch.Tensor
    y: torch.Tensor


@dataclasses.dataclass
class _Input:
    x: torch.Tensor


@dataclasses.dataclass
class _Transfer:
    hidden: torch.Tensor


@dataclasses.dataclass
class _State:
    y: torch.Tensor


class _RegressionDataset(Dataset):
    def __init__(self, num_samples: int):
        # The loop seeds the global generator of the data provider, so every run builds the same data.
        self._x = torch.randn(num_samples, _NUM_FEATURES)
        self._y = self._x @ torch.randn(_NUM_FEATURES, 1)

    def __getitem__(self, index: int) -> _Sample:
        return _Sample(x=self._x[index], y=self._y[index])

    def __len__(self) -> int:
        return len(self._x)

    @staticmethod
    def collate(samples: Sequence[_Sample]) -> _Sample:
        return _Sample(x=torch.stack([s.x for s in samples]), y=torch.stack([s.y for s in samples]))


class _StageMLP(nn.Module, ModuleSupportsPipelining):
    # The first stage holds the input layer and the last stage the output layer, so one stage holds both.
    def __init__(self, stage: PipelineStageInfo):
        super().__init__()
        self._stage = stage
        if stage.is_current_stage_first:
            self.up = nn.Linear(_NUM_FEATURES, _HIDDEN_SIZE)
        if stage.is_current_stage_last:
            self.down = nn.Linear(_HIDDEN_SIZE, 1)

    def forward(self, inputs: _Input | _Transfer, shared: None) -> _Transfer | torch.Tensor:
        hidden = torch.relu(self.up(inputs.x)) if self._stage.is_current_stage_first else inputs.hidden
        if self._stage.is_current_stage_last:
            return self.down(hidden)
        return _Transfer(hidden=hidden)

    def stage_transfer_spec(self, pipeline_input: _Input, boundary: StageBoundary) -> _Transfer:
        return _Transfer(hidden=TensorSpec(shape=(pipeline_input.x.shape[0], _HIDDEN_SIZE), dtype=torch.float32))

    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()


class _MLPProvider(ModelProvider[_StageMLP]):
    """Builds the MLP stages and records the steps it sees, optionally sleeping and waking the trainer after one."""

    def __init__(self, sleep_after_step: int | None):
        self._sleep_after_step = sleep_after_step
        self.trainer: Trainer | None = None
        self.steps: list[int] = []
        self.slept_after_steps: list[int] = []

    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult[_StageMLP]:
        model = _StageMLP(context.stage)
        return InitializeModelStageResult(model=model, state_mapper=identity_mapper_from_module(model))

    def parallelize_model_stage(self, context: ParallelizeModelStageContext[_StageMLP]):
        mesh = context.dist_context.mesh_for(DENSE_DOMAIN)
        parallelize_hsdp(context.model, mesh=mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])

    def prepare_export_model_stage(
        self, context: PrepareExportModelStageContext[_StageMLP]
    ) -> PrepareExportModelStageResult:
        return PrepareExportModelStageResult(state_mapper=identity_mapper_from_module(context.model))

    def register_events(self, context: RegisterModelEventsContext) -> None:
        context.event_bus.subscribe(EVENT_TRAIN_STEP_POST, self._on_step_post)

    def _on_step_post(self, ctx: EventStepContext):
        step = ctx.schedule.current_step
        self.steps.append(step)
        if step == self._sleep_after_step:
            self.trainer.sleep()
            if self.trainer.is_sleeping(SleepTag.TENSOR_STATES):
                self.slept_after_steps.append(step)
            self.trainer.wake()


class _RegressionTask(TrainTask[_Sample, _Input, None, torch.Tensor, _State]):
    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext[_Sample]
    ) -> BuildForwardInputsResult[_Input, None, _State]:
        return BuildForwardInputsResult(input=_Input(x=ctx.batch.x), shared=None, state=_State(y=ctx.batch.y))

    def compute_loss(self, ctx: ComputeLossContext[torch.Tensor, _State]) -> ComputeLossResult:
        return ComputeLossResult(loss=nn.functional.mse_loss(ctx.pipeline_results, ctx.state.y), loss_weight=None)


def _train(
    mesh: DeviceMeshParameters, save_dir: Path, total_steps: int, sleep_after_step: int | None = None
) -> tuple[Trainer, _MLPProvider]:
    config = TrainerConfig.model_validate(
        {
            "run": {"name": "runtime_loop"},
            "schedule": {"total_steps": total_steps},
            "logging": {"period_steps": 1, "tracker": {"provider": "null"}},
            "determinism": {"base_seed": 42},
            "gc": {"period_steps": 100},
            "checkpointing": {"save_dir": save_dir, "period_steps": _HALF_STEPS},
            "gradient_clipping": {"max_norm": 1.0},
        }
    )
    provider = _MLPProvider(sleep_after_step)
    trainer = TrainingConfigurator(
        mesh=mesh,
        parameters=config,
        task_provider=lambda ctx: _RegressionTask(),
        model_provider=provider,
        data_provider=AutoDataProvider(
            dataset_factory=lambda context: _RegressionDataset(num_samples=4096),
            collator=_RegressionDataset.collate,
            config=AutoDataConfig(global_batch_size=256, microbatch_size=16, num_workers=0, shuffle=True),
        ),
        optimizer_provider=AutoOptimizerProvider(
            TypeAdapter(AutoOptimizerConfig).validate_python({"name": "adamw", "lr": 0.01})
        ),
        # A constant learning rate, so a run of half the steps follows the same path as the full run.
        lr_scheduler_provider=AutoLRSchedulerProvider(
            TypeAdapter(AutoLRSchedulerConfig).validate_python(
                {
                    "name": "piecewise",
                    "scheduler": {
                        "initial_multiplier": 1.0,
                        "phases": [{"mode": "rest", "target_multiplier": 1.0, "curve": {"type": "linear"}}],
                    },
                }
            )
        ),
    ).configure()
    provider.trainer = trainer
    trainer.train()
    return trainer, provider


def _exported_weights(trainer: Trainer, export_dir: Path, wait_world) -> dict[str, torch.Tensor]:
    trainer.export(export_dir, load_checkpoint=False)
    wait_world()
    weights = {}
    for file in sorted(export_dir.glob("*.safetensors")):
        weights.update(load_file(file))
    return weights


def _assert_weights_equal(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]):
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name


@pytest.mark.distributed
@pytest.mark.parametrize(
    "mesh",
    [
        DeviceMeshParameters(data_parallel_shard=8),
        DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4),
        DeviceMeshParameters(pipeline_parallel=2, data_parallel_shard=4),
    ],
    ids=["fsdp", "hsdp", "pp_fsdp"],
)
def test_sleep_and_resume_match_plain_run(mesh, dist_ctx_factory, shared_tmp_dir):
    wait_world = dist_ctx_factory(mesh).wait_world

    plain_trainer, plain = _train(mesh, shared_tmp_dir / "plain", _TOTAL_STEPS)
    expected = _exported_weights(plain_trainer, shared_tmp_dir / "plain_export", wait_world)
    assert plain.steps == list(range(_TOTAL_STEPS))
    assert expected

    # A sleep and a wake in the middle of the run move the weights and the optimizer state off the GPU and back.
    sleep_trainer, sleeping = _train(mesh, shared_tmp_dir / "sleep", _TOTAL_STEPS, sleep_after_step=_HALF_STEPS)
    assert sleeping.slept_after_steps == [_HALF_STEPS]
    _assert_weights_equal(_exported_weights(sleep_trainer, shared_tmp_dir / "sleep_export", wait_world), expected)

    # A job that stops after half the steps leaves a checkpoint, and a new job resumes from it.
    _train(mesh, shared_tmp_dir / "resume", _HALF_STEPS)
    resume_trainer, resumed = _train(mesh, shared_tmp_dir / "resume", _TOTAL_STEPS)
    assert resumed.steps == list(range(_HALF_STEPS, _TOTAL_STEPS))
    _assert_weights_equal(_exported_weights(resume_trainer, shared_tmp_dir / "resume_export", wait_world), expected)
