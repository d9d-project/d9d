from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import datasets
import torch
from d9d.core.dist_context import DeviceMeshParameters, DistributedContext
from d9d.core.types import ScalarTree, TensorTree
from d9d.dataset import (
    BufferSortedDataset,
    DatasetImplementingSortKeyProtocol,
    pad_stack_1d,
)
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
    CreateMetricsContext,
    CreateMetricsResult,
    InitializeModelStageContext,
    InitializeModelStageResult,
    ModelProvider,
    ParallelizeModelStageContext,
    PrepareExportModelStageContext,
    PrepareExportModelStageResult,
    TrainTask,
    UpdateMetricsContext,
)
from d9d.loop.run import TrainingConfigurator
from d9d.metric.impl.aggregation import SumMetric
from d9d.model_state.mapper.adapters import identity_mapper_from_module
from d9d.module.block.head import LM_IGNORE_INDEX, SequenceCausalLMHeadShared, SequenceCausalLMOutput
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model import DecoderForCausalLM
from d9d.module.model.io import SequenceHeadShared, SequenceInput, SequenceShared
from d9d.module.model.qwen3_moe import Qwen3MoEModel, Qwen3MoEParameters
from d9d.module.parallelism.model import parallelize_causal_lm_head, parallelize_qwen3_moe_model
from pydantic import BaseModel
from tokenizers import Tokenizer
from torch.utils.data import Dataset

# -----------------------------------
# Configuration Schema using Pydantic
# -----------------------------------


class DataConfig(BaseModel):
    dataset: str  # Hugging Face dataset path/name
    split: str  # E.g. 'train', 'validation'
    text_column: str  # The column containing the raw text
    use_samples: int  # Limit dataset size for testing/debugging
    shuffle_seed: int  # Distinct seed for shuffling the data
    tokenizer: str  # Path to the tokenizer.json file
    num_proc: int  # Number of CPU processes for data mapping
    presort_buffer_size: int  # Size of buffer for length-based presorting
    presort_pack_size: int  # Window (in samples) the buffer sorts by length before yielding


class ModelProviderConfig(BaseModel):
    model: Qwen3MoEParameters  # Hyperparameters for the Qwen3 MoE backbone
    checkpointing: bool  # Enable activation checkpointing to save GPU memory


class ProjectConfig(BaseModel):
    data: DataConfig
    auto_data: AutoDataConfig  # Batch sizing + DataLoader settings for the default data stack
    mesh: DeviceMeshParameters
    model_provider: ModelProviderConfig
    trainer: TrainerConfig
    optimizer: AutoOptimizerConfig
    lr_scheduler: AutoLRSchedulerConfig
    export_to: Path  # Directory to save the final model


# ----------------------
# Dataset Implementation
# ----------------------


class ProjectDataset(Dataset, DatasetImplementingSortKeyProtocol):
    def __init__(self, dataset: datasets.Dataset, tokenizer: Tokenizer):
        self._dataset = dataset
        self._tokenizer = tokenizer

    def sort_key(self, index: int) -> Any:
        # Used by BufferSortedDataset to group examples of similar length together.
        # This minimizes padding overhead in batches.
        return self._dataset[index]["token_counts"]

    def __getitem__(self, index: int) -> TensorTree:
        item = self._dataset[index]
        tokens = torch.tensor(self._tokenizer.encode(item["text"]).ids, dtype=torch.long)

        # Standard causal LM shift:
        # Input: [A, B, C]
        # Label: [B, C, D]
        # d9d models do not shift labels, to avoid extra GPU work. Shift them here, on the CPU.
        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Position IDs are 0..N-1.
        position_ids = torch.arange(0, input_ids.shape[0], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}

    @classmethod
    def collate(cls, batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            # Pad inputs to max length in this batch with 0.
            "input_ids": pad_stack_1d([x["input_ids"] for x in batch], pad_value=0),
            # Pad labels with LM_IGNORE_INDEX (-100), which the loss ignores.
            "labels": pad_stack_1d([x["labels"] for x in batch], pad_value=LM_IGNORE_INDEX),
            # Pad position IDs.
            "position_ids": pad_stack_1d([x["position_ids"] for x in batch], pad_value=0),
        }

    def __len__(self) -> int:
        return len(self._dataset)


def _count_tokens(item: dict, text_column: str, tokenizer: Tokenizer) -> dict:
    return {
        "token_counts": len(tokenizer.encode(item[text_column]).tokens),
    }


def build_dataset(config: DataConfig, dist_context: DistributedContext) -> Dataset:
    tokenizer = Tokenizer.from_file(str(config.tokenizer))

    # main_process_first lets rank 0 download and process the dataset and fill the cache first.
    # The other ranks wait, then load from the cache. This avoids races on the Hugging Face cache.
    with dist_context.main_process_first():
        data = (
            datasets.load_dataset(config.dataset, split=config.split)
            .take(config.use_samples)
            .shuffle(config.shuffle_seed)
            .map(
                _count_tokens,
                num_proc=config.num_proc,
                fn_kwargs={"tokenizer": tokenizer, "text_column": config.text_column},
            )
        )

    dataset = ProjectDataset(data, tokenizer)

    # BufferSortedDataset groups samples of similar length (see sort_key above) and shuffles the groups.
    # This minimizes padding. Return it unsharded: AutoDataProvider shards it across data-parallel ranks.
    return BufferSortedDataset(
        dataset,
        buffer_size=config.presort_buffer_size,
        pack_size=config.presort_pack_size,
        init_seed=config.shuffle_seed,
    )


# --------------
# Model Provider
# --------------


class ProjectModelProvider(ModelProvider[DecoderForCausalLM[Qwen3MoEModel]]):
    def __init__(self, config: ModelProviderConfig):
        self._config = config

    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult:
        # The loop calls this method on the meta device, so no weights are allocated here.
        # Compose the Qwen3 MoE backbone with a single causal LM head and cast it to bf16.
        backbone = Qwen3MoEModel(
            params=self._config.model,
            stage=context.stage,
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=self._config.checkpointing,
        )
        model = DecoderForCausalLM(backbone, context.stage).bfloat16()

        return InitializeModelStageResult(
            model=model,
            state_mapper=identity_mapper_from_module(model),
        )

    def parallelize_model_stage(self, context: ParallelizeModelStageContext):
        # Apply the Qwen3 MoE parallelism routine to the backbone and the head routine to the head.
        # You can apply your own horizontal parallelism strategy here.
        parallelize_qwen3_moe_model(context.dist_context, context.model.model, context.stage)
        if context.stage.is_current_stage_last:
            parallelize_causal_lm_head(context.model.head, context.dist_context)

    def prepare_export_model_stage(self, context: PrepareExportModelStageContext) -> PrepareExportModelStageResult:
        # When exporting, save model weights as-is.

        return PrepareExportModelStageResult(state_mapper=identity_mapper_from_module(context.model))

    def dump_hparams(self) -> ScalarTree:
        return self._config.model_dump(mode="json")


# --------------
# Training Logic
# --------------


class SFTState(TypedDict):
    # Side-data carried from build_forward_inputs to loss/metric computation for the same microbatch.
    labels: torch.Tensor
    num_tokens: torch.Tensor


class SFTTask(
    TrainTask[
        dict[str, torch.Tensor],
        SequenceInput,
        SequenceHeadShared[SequenceCausalLMHeadShared],
        SequenceCausalLMOutput,
        SFTState,
    ]
):
    def __init__(self, dist_ctx: DistributedContext):
        self._dist_ctx = dist_ctx

    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext
    ) -> BuildForwardInputsResult[SequenceInput, SequenceHeadShared[SequenceCausalLMHeadShared], SFTState]:
        # ctx.batch contains the output of the collator.

        labels = ctx.batch["labels"]

        # Count the valid tokens (labels that are not padding). The loss of a variable-length batch needs it.
        num_tokens = (labels != LM_IGNORE_INDEX).sum()

        # Return the pipeline input (first stage only) plus the shared input (every stage) and the
        # typed side-data carried to loss/metric computation. The shared input routes position ids
        # to the backbone and labels to the head.
        return BuildForwardInputsResult(
            input=SequenceInput(input_ids=ctx.batch["input_ids"]),
            shared=SequenceHeadShared(
                sequence=SequenceShared(position_ids=ctx.batch["position_ids"]),
                head=SequenceCausalLMHeadShared(labels=ctx.batch["labels"]),
            ),
            state=SFTState(labels=labels, num_tokens=num_tokens),
        )

    def dump_hparams(self) -> ScalarTree:
        return super().dump_hparams()

    def create_metrics(self, ctx: CreateMetricsContext) -> CreateMetricsResult:
        return CreateMetricsResult(
            metrics={
                "num_tokens": SumMetric(),
            }
        )

    def update_metrics(self, ctx: UpdateMetricsContext[SFTState]):
        ctx.metrics["num_tokens"].update(ctx.state["num_tokens"])

    def compute_loss(self, ctx: ComputeLossContext[SequenceCausalLMOutput, SFTState]) -> ComputeLossResult:
        # Retrieve log_probs calculated by the model pipeline.
        logps = ctx.pipeline_results.logps

        num_loss_tokens = ctx.state["num_tokens"]

        # Calculate average loss per valid token.
        total_loss = logps.sum() / num_loss_tokens

        return ComputeLossResult(
            loss=total_loss,
            # loss_weight weights this microbatch in the gradient average across microbatches and ranks.
            # Weighting by token count gives the true per-token average when token counts differ.
            loss_weight=num_loss_tokens / 1000,
        )


# ---------------------
# Execution Entry Point
# ---------------------


def main():
    # 1. Load Configuration
    # Uses Pydantic to validate the JSON structure against the class definitions above.
    config = ProjectConfig.model_validate_json(Path("pretrain.json").read_text(encoding="utf-8"))

    # 2. Dependency Injection / Construction
    # TrainingConfigurator assembles the distributed environment, the model, the optimizer and the data stream.
    trainer = TrainingConfigurator(
        mesh=config.mesh,
        parameters=config.trainer,
        task_provider=lambda ctx: SFTTask(ctx.dist_context),
        model_provider=ProjectModelProvider(config.model_provider),
        data_provider=AutoDataProvider(
            dataset_factory=lambda dist_context: build_dataset(config.data, dist_context),
            collator=ProjectDataset.collate,
            config=config.auto_data,
        ),
        optimizer_provider=AutoOptimizerProvider(config.optimizer),
        lr_scheduler_provider=AutoLRSchedulerProvider(config.lr_scheduler),
    ).configure()

    # 3. Execution
    trainer.train()

    # 4. Finalization - export the trained model to disk
    trainer.export(config.export_to, load_checkpoint=False)


if __name__ == "__main__":
    main()
