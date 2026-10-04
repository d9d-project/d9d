import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from d9d.core.dist_context import DENSE_DOMAIN, DeviceMeshParameters
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
    TrainTask,
)
from d9d.loop.run import TrainingConfigurator
from d9d.model_state.mapper.adapters import identity_mapper_from_module
from d9d.module.parallelism.api import parallelize_hsdp
from pydantic import BaseModel
from torch import nn
from torch.utils.data import Dataset


class ProjectConfig(BaseModel):
    mesh: DeviceMeshParameters
    data: AutoDataConfig
    trainer: TrainerConfig
    optimizer: AutoOptimizerConfig
    lr_scheduler: AutoLRSchedulerConfig
    export_to: Path


# One sample of the dataset.
@dataclasses.dataclass
class RegressionSample:
    x: torch.Tensor
    y: torch.Tensor


# One microbatch, as collate() builds it.
@dataclasses.dataclass
class RegressionBatch:
    x: torch.Tensor
    y: torch.Tensor


# What the model receives.
@dataclasses.dataclass
class RegressionInput:
    x: torch.Tensor


# What compute_loss() needs but the model does not receive.
@dataclasses.dataclass
class RegressionState:
    y: torch.Tensor


# A synthetic regression task: the targets are a fixed random linear function of the inputs plus noise.
class RegressionDataset(Dataset):
    def __init__(self, num_samples: int, num_features: int):
        generator = torch.Generator().manual_seed(0)
        self._x = torch.randn(num_samples, num_features, generator=generator)
        true_weight = torch.randn(num_features, 1, generator=generator)
        self._y = self._x @ true_weight + 0.1 * torch.randn(num_samples, 1, generator=generator)

    def __getitem__(self, index: int) -> RegressionSample:
        return RegressionSample(x=self._x[index], y=self._y[index])

    def __len__(self) -> int:
        return len(self._x)

    @staticmethod
    def collate(samples: Sequence[RegressionSample]) -> RegressionBatch:
        return RegressionBatch(x=torch.stack([s.x for s in samples]), y=torch.stack([s.y for s in samples]))


# A plain PyTorch model. d9d needs two things from it: reset_parameters() for initialization on the
# device, and forward(inputs, shared), which is the signature of a pipeline stage.
class MLP(nn.Module):
    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        self.up = nn.Linear(num_features, hidden_size)
        self.down = nn.Linear(hidden_size, 1)

    def forward(self, inputs: RegressionInput, shared: None) -> torch.Tensor:
        return self.down(torch.relu(self.up(inputs.x)))

    def reset_parameters(self):
        self.up.reset_parameters()
        self.down.reset_parameters()


class MLPProvider(ModelProvider[MLP]):
    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult[MLP]:
        model = MLP(num_features=16, hidden_size=64)
        return InitializeModelStageResult(model=model, state_mapper=identity_mapper_from_module(model))

    def parallelize_model_stage(self, context: ParallelizeModelStageContext[MLP]):
        # HSDP: shards along dp_cp_shard and replicates along the other dimensions. Size-1 dimensions are skipped.
        mesh = context.dist_context.mesh_for(DENSE_DOMAIN)
        parallelize_hsdp(context.model, mesh=mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])

    def prepare_export_model_stage(self, context: PrepareExportModelStageContext[MLP]) -> PrepareExportModelStageResult:
        return PrepareExportModelStageResult(state_mapper=identity_mapper_from_module(context.model))


class RegressionTask(TrainTask[RegressionBatch, RegressionInput, None, torch.Tensor, RegressionState]):
    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext[RegressionBatch]
    ) -> BuildForwardInputsResult[RegressionInput, None, RegressionState]:
        # The model gets the inputs. The targets wait in the state until compute_loss().
        return BuildForwardInputsResult(
            input=RegressionInput(x=ctx.batch.x), shared=None, state=RegressionState(y=ctx.batch.y)
        )

    def compute_loss(self, ctx: ComputeLossContext[torch.Tensor, RegressionState]) -> ComputeLossResult:
        loss = nn.functional.mse_loss(ctx.pipeline_results, ctx.state.y)
        return ComputeLossResult(loss=loss, loss_weight=None)


def main():
    config = ProjectConfig.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")))

    trainer = TrainingConfigurator(
        mesh=config.mesh,
        parameters=config.trainer,
        task_provider=lambda ctx: RegressionTask(),
        model_provider=MLPProvider(),
        data_provider=AutoDataProvider(
            dataset_factory=lambda dist_context: RegressionDataset(num_samples=65536, num_features=16),
            collator=RegressionDataset.collate,
            config=config.data,
        ),
        optimizer_provider=AutoOptimizerProvider(config.optimizer),
        lr_scheduler_provider=AutoLRSchedulerProvider(config.lr_scheduler),
    ).configure()

    trainer.train()
    trainer.export(config.export_to, load_checkpoint=False)


if __name__ == "__main__":
    main()
