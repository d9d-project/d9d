import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

import datasets
import torch
import yaml
from d9d.core.dist_context import DeviceMeshParameters, DistributedContext
from d9d.dataset import pad_stack_1d
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
from d9d.model_state.mapper.compose import ModelStateMapperSequential
from d9d.module.block.head import LM_IGNORE_INDEX, SequenceCausalLMHeadShared, SequenceCausalLMOutput
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model import DecoderForCausalLM
from d9d.module.model.io import SequenceHeadShared, SequenceInput, SequenceShared
from d9d.module.model.qwen3_dense import (
    Qwen3DenseModel,
    Qwen3DenseParameters,
    mapper_from_huggingface_qwen3_dense_for_causal_lm,
    mapper_to_huggingface_qwen3_dense_for_causal_lm,
)
from d9d.module.parallelism.model import parallelize_causal_lm_head, parallelize_qwen3_dense_model
from d9d.peft import inject_peft_and_freeze, merge_peft
from d9d.peft.lora import LoRA, LoRAConfig
from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer
from torch.utils.data import Dataset


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    dataset_config: str | None
    split: str
    text_column: str
    tokenizer: Path
    max_length: int


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mesh: DeviceMeshParameters
    model: Qwen3DenseParameters
    lora: LoRAConfig | None
    data: DataConfig
    auto_data: AutoDataConfig
    trainer: TrainerConfig
    optimizer: AutoOptimizerConfig
    lr_scheduler: AutoLRSchedulerConfig
    export_to: Path


# One microbatch of token sequences, padded to one length.
@dataclasses.dataclass
class TextBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Tokenizer, max_length: int):
        self._samples = []
        for text in texts:
            # One more token than max_length, because the labels are the inputs shifted by one.
            ids = tokenizer.encode(text).ids[: max_length + 1]
            if len(ids) > 1:
                self._samples.append(torch.tensor(ids, dtype=torch.long))

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def collate(samples: Sequence[torch.Tensor]) -> TextBatch:
        # d9d models do not shift the labels, so shift them here.
        return TextBatch(
            input_ids=pad_stack_1d([tokens[:-1] for tokens in samples], pad_value=0),
            labels=pad_stack_1d([tokens[1:] for tokens in samples], pad_value=LM_IGNORE_INDEX),
            position_ids=pad_stack_1d([torch.arange(len(tokens) - 1) for tokens in samples], pad_value=0),
        )


def build_dataset(config: DataConfig, dist_context: DistributedContext) -> Dataset:
    # Rank 0 downloads the dataset first. The other ranks then read it from the cache.
    with dist_context.main_process_first():
        data = datasets.load_dataset(config.dataset, config.dataset_config, split=config.split)
    texts = [text for text in data[config.text_column] if text.strip()]
    return TextDataset(texts, Tokenizer.from_file(str(config.tokenizer)), config.max_length)


# --8<-- [start:provider]
class Qwen3Provider(ModelProvider[DecoderForCausalLM[Qwen3DenseModel]]):
    def __init__(self, params: Qwen3DenseParameters, lora: LoRAConfig | None):
        self._params = params
        self._lora = LoRA(lora) if lora is not None else None

    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult:
        backbone = Qwen3DenseModel(
            params=self._params,
            stage=context.stage,
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=True,
        )
        model = DecoderForCausalLM(backbone, context.stage).bfloat16()

        # Maps the Hugging Face checkpoint keys to the d9d model keys.
        state_mapper = mapper_from_huggingface_qwen3_dense_for_causal_lm(self._params)

        if self._lora is not None:
            # Wraps the matching layers in LoRA adapters and freezes everything else.
            # The LoRA mapper then moves the original weights of the wrapped layers to their new keys.
            lora_mapper = inject_peft_and_freeze(self._lora, model)
            state_mapper = ModelStateMapperSequential([state_mapper, lora_mapper])

        return InitializeModelStageResult(model=model, state_mapper=state_mapper)

    def parallelize_model_stage(self, context: ParallelizeModelStageContext):
        parallelize_qwen3_dense_model(context.dist_context, context.model.model, context.stage)
        if context.stage.is_current_stage_last:
            parallelize_causal_lm_head(context.model.head, context.dist_context)

    def prepare_export_model_stage(self, context: PrepareExportModelStageContext) -> PrepareExportModelStageResult:
        if self._lora is not None:
            # Merges the adapters into the base weights, so the export has the original architecture.
            merge_peft(self._lora, context.model)

        # Exports the model in the Hugging Face format.
        return PrepareExportModelStageResult(state_mapper=mapper_to_huggingface_qwen3_dense_for_causal_lm(self._params))


# --8<-- [end:provider]


# --8<-- [start:task]
@dataclasses.dataclass
class CausalLMState:
    num_tokens: torch.Tensor


class CausalLMTask(
    TrainTask[
        TextBatch,
        SequenceInput,
        SequenceHeadShared[SequenceCausalLMHeadShared],
        SequenceCausalLMOutput,
        CausalLMState,
    ]
):
    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext[TextBatch]
    ) -> BuildForwardInputsResult[SequenceInput, SequenceHeadShared[SequenceCausalLMHeadShared], CausalLMState]:
        labels = ctx.batch.labels
        return BuildForwardInputsResult(
            input=SequenceInput(input_ids=ctx.batch.input_ids),
            shared=SequenceHeadShared(
                sequence=SequenceShared(position_ids=ctx.batch.position_ids),
                head=SequenceCausalLMHeadShared(labels=labels),
            ),
            state=CausalLMState(num_tokens=(labels != LM_IGNORE_INDEX).sum()),
        )

    def compute_loss(self, ctx: ComputeLossContext[SequenceCausalLMOutput, CausalLMState]) -> ComputeLossResult:
        # The head returns the per-token cross-entropy loss. Padding positions hold zeros.
        num_tokens = ctx.state.num_tokens
        loss = ctx.pipeline_results.logps.sum() / num_tokens

        # Weighting each microbatch by its token count gives the mean loss per token across microbatches and ranks.
        return ComputeLossResult(loss=loss, loss_weight=num_tokens.float())


# --8<-- [end:task]


def main():
    config = ProjectConfig.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")))

    trainer = TrainingConfigurator(
        mesh=config.mesh,
        parameters=config.trainer,
        task_provider=lambda ctx: CausalLMTask(),
        model_provider=Qwen3Provider(config.model, config.lora),
        data_provider=AutoDataProvider(
            dataset_factory=lambda dist_context: build_dataset(config.data, dist_context),
            collator=TextDataset.collate,
            config=config.auto_data,
        ),
        optimizer_provider=AutoOptimizerProvider(config.optimizer),
        lr_scheduler_provider=AutoLRSchedulerProvider(config.lr_scheduler),
    ).configure()

    trainer.train()
    trainer.export(config.export_to, load_checkpoint=False)


if __name__ == "__main__":
    main()
