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
from d9d.loop.auto import AutoDataConfig, AutoDataProvider
from d9d.loop.config import InferenceConfig
from d9d.loop.control import (
    BuildForwardInputsContext,
    BuildForwardInputsResult,
    InferenceTask,
    InitializeModelStageContext,
    InitializeModelStageResult,
    ModelProvider,
    ParallelizeModelStageContext,
    PrepareExportModelStageContext,
    PrepareExportModelStageResult,
    ProcessOutputsContext,
)
from d9d.loop.run import InferenceConfigurator
from d9d.model_state.mapper.adapters import identity_mapper_from_module
from d9d.module.block.head import LM_IGNORE_INDEX
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model.qwen3_moe import Qwen3MoEForCausalLM, Qwen3MoEForCausalLMParameters
from d9d.module.parallelism.model.qwen3_moe import parallelize_qwen3_moe_for_causal_lm
from pydantic import BaseModel
from tokenizers import Tokenizer
from torch.utils.data import Dataset

# -----------------------------------
# Configuration Schema using Pydantic
# -----------------------------------


class DataConfig(BaseModel):
    dataset: str  # HuggingFace dataset path/name
    split: str  # e.g., 'train', 'validation'
    text_column: str  # The column containing the raw text
    use_samples: int  # Limit dataset size for testing/debugging
    shuffle_seed: int  # Distinct seed for shuffling the data
    tokenizer: str  # Path to the tokenizer.json file
    num_proc: int  # Number of CPU processes for data mapping
    presort_buffer_size: int  # Size of buffer for length-based presorting
    presort_pack_size: int  # Window (in samples) the buffer sorts by length before yielding


class ModelProviderConfig(BaseModel):
    model: Qwen3MoEForCausalLMParameters  # Hyperparameters for Qwen3 MoE
    checkpointing: bool  # Enable gradient checkpointing to save VRAM


class ProjectConfig(BaseModel):
    data: DataConfig
    auto_data: AutoDataConfig  # Batch sizing + DataLoader settings for the default data stack
    mesh: DeviceMeshParameters
    model_provider: ModelProviderConfig
    inference: InferenceConfig


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
        # Encode text to tokens
        tokens = torch.tensor(self._tokenizer.encode(item["text"]).ids, dtype=torch.long)

        # Standard Causal LM logic:
        # Input: [A, B, C]
        # Label: [B, C, D]
        # d9d models do NOT handle this logic to not introduce additional GPU overhead, so we do this in data
        # processing on CPU:
        input_ids = tokens[:-1]
        labels = tokens[1:]

        # Position IDs usually 0..N-1
        position_ids = torch.arange(0, input_ids.shape[0], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }

    @classmethod
    def collate(cls, batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            # Pad inputs to max length in this batch with 0
            "input_ids": pad_stack_1d([x["input_ids"] for x in batch], pad_value=0),
            # Pad labels with -100 (we ignore this value by default)
            "labels": pad_stack_1d([x["labels"] for x in batch], pad_value=LM_IGNORE_INDEX),
            # Pad position tokens
            "position_ids": pad_stack_1d([x["position_ids"] for x in batch], pad_value=0),
        }

    def __len__(self) -> int:
        return len(self._dataset)


def _count_tokens(item: dict, text_column: str, tokenizer: Tokenizer) -> dict:
    return {
        "token_counts": len(tokenizer.encode(item[text_column]).tokens),
    }


def build_dataset(config: DataConfig, dist_context: DistributedContext) -> Dataset:
    """Builds the (unsharded) dataset. AutoDataProvider handles sharding, loading, and packing.

    Args:
        config: The data configuration.
        dist_context: The distributed context (used to guard dataset preparation on the main process).

    Returns:
        The unsharded, length-bucketed dataset.
    """
    tokenizer = Tokenizer.from_file(str(config.tokenizer))

    # IMPORTANT: main_process_first ensures that Rank 0 downloads/processes
    # the dataset and builds the cache first. Ranks 1-N wait, then load from cache.
    # Prevents race conditions and corruption on the HF cache.
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

    # BufferSortedDataset acts as a buffer that shuffles data locally
    # but outputs batches sorted by length (defined in sort_key above) to minimize padding overhead.
    # Return it UNSHARDED - AutoDataProvider shards across data-parallel ranks itself.
    return BufferSortedDataset(
        dataset,
        buffer_size=config.presort_buffer_size,
        pack_size=config.presort_pack_size,
        init_seed=config.shuffle_seed,
    )


# --------------
# Model Provider
# --------------


class ProjectModelProvider(ModelProvider[Qwen3MoEForCausalLM]):
    def __init__(self, config: ModelProviderConfig):
        self._config = config

    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult:
        # Initialize the raw model on CPU or Meta device in BF16 precision
        model = Qwen3MoEForCausalLM(
            params=self._config.model,
            stage=context.stage,
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=self._config.checkpointing,
        ).bfloat16()

        return InitializeModelStageResult(
            model=model,
            state_mapper=identity_mapper_from_module(model),
        )

    def parallelize_model_stage(self, context: ParallelizeModelStageContext):
        # Applies specific distributed strategies
        # suited for Qwen3 MoE architecture.
        # You can apply your own horizontal parallelism strategy here.
        parallelize_qwen3_moe_for_causal_lm(dist_context=context.dist_context, stage=context.stage, model=context.model)

    def prepare_export_model_stage(self, context: PrepareExportModelStageContext) -> PrepareExportModelStageResult:
        # When exporting, save model weights as-is

        return PrepareExportModelStageResult(state_mapper=identity_mapper_from_module(context.model))

    def dump_hparams(self) -> ScalarTree:
        return self._config.model_dump(mode="json")


# --------------
# Inference Logic
# --------------


class PerplexityState(TypedDict):
    # Side-data carried from build_forward_inputs to output processing for the same microbatch.
    labels: torch.Tensor


class PerplexityTask(InferenceTask[dict[str, torch.Tensor], PerplexityState]):
    def __init__(self, dist_ctx: DistributedContext):
        self._dist_ctx = dist_ctx
        self._cache: list[torch.Tensor] = []

    def build_forward_inputs(self, ctx: BuildForwardInputsContext) -> BuildForwardInputsResult[PerplexityState]:
        # ctx.batch contains the output of the Collator.

        # Return inputs for model.forward() plus the typed side-data for output processing.
        # inputs are only for the first pipeline stage
        # kwargs are the same for all the pipeline stages
        return BuildForwardInputsResult(
            inputs={
                "input_ids": ctx.batch["input_ids"],
            },
            kwargs={
                "labels": ctx.batch["labels"],
                "position_ids": ctx.batch["position_ids"],
            },
            state=PerplexityState(labels=ctx.batch["labels"]),
        )

    def process_outputs(self, ctx: ProcessOutputsContext[PerplexityState]):
        logps = ctx.pipeline_results["logps"]

        # Calculate number of valid tokens (ignoring the -100 padding)
        # This is crucial for variable length batches.
        num_loss_tokens = (ctx.state["labels"] != LM_IGNORE_INDEX).sum()

        # Calculate average loss per valid token
        perplexity = logps.sum() / num_loss_tokens

        self._cache.append(perplexity)

        if len(self._cache) == 100:
            mean = torch.stack(self._cache).sum().item()
            self._dist_ctx.logger.info(f"Processed 100 samples, mean perplexity: {mean}")
            self._cache.clear()


# ---------------------
# Execution Entry Point
# ---------------------


def main():
    # 1. Load Configuration
    # Uses Pydantic to validate the JSON structure against the class definitions above.
    config = ProjectConfig.model_validate_json(Path("calculate_perplexity.json").read_text(encoding="utf-8"))

    # 2. Dependency Injection / Construction
    inference = InferenceConfigurator(
        mesh=config.mesh,
        parameters=config.inference,
        task_provider=lambda ctx: PerplexityTask(ctx.dist_context),
        model_provider=ProjectModelProvider(config.model_provider),
        data_provider=AutoDataProvider(
            dataset_factory=lambda dist_context: build_dataset(config.data, dist_context),
            collator=ProjectDataset.collate,
            config=config.auto_data,
        ),
    ).configure()

    # 3. Execution
    inference.infer()


if __name__ == "__main__":
    main()
