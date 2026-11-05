import dataclasses
import logging
import math
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from d9d.dist_context import METRIC_DOMAIN, DeviceMeshParameters, DistributedContext, MetricDim, init_dist_context
from d9d.loop.auto import AutoDataConfig, AutoDataProvider
from d9d.loop.config import InferenceConfig
from d9d.loop.control import (
    BuildForwardInputsContext,
    BuildForwardInputsResult,
    FinalizeContext,
    InferenceTask,
    ProcessOutputsContext,
)
from d9d.loop.run import InferenceConfigurator
from d9d.models.io import SequenceHeadShared, SequenceInput, SequenceShared
from d9d.models.qwen3_dense import Qwen3DenseParameters
from d9d.nn.head import LM_IGNORE_INDEX, SequenceCausalLMHeadShared, SequenceCausalLMOutput
from finetune import DataConfig, Qwen3Provider, TextBatch, TextDataset, build_dataset
from pydantic import BaseModel, ConfigDict

# A child of the d9d logger, so its lines get the rank prefix that the configurator sets up.
logger = logging.getLogger("d9d.evaluate")


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mesh: DeviceMeshParameters
    model: Qwen3DenseParameters
    data: DataConfig
    auto_data: AutoDataConfig
    inference: InferenceConfig


# --8<-- [start:task]
@dataclasses.dataclass
class PerplexityState:
    num_tokens: torch.Tensor


class PerplexityTask(
    InferenceTask[
        TextBatch,
        SequenceInput,
        SequenceHeadShared[SequenceCausalLMHeadShared],
        SequenceCausalLMOutput,
        PerplexityState,
    ]
):
    def __init__(self, dist_context: DistributedContext):
        self._dist_context = dist_context
        self._loss_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        self._num_tokens = torch.zeros((), dtype=torch.float64, device="cuda")

    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext[TextBatch]
    ) -> BuildForwardInputsResult[SequenceInput, SequenceHeadShared[SequenceCausalLMHeadShared], PerplexityState]:
        labels = ctx.batch.labels
        return BuildForwardInputsResult(
            input=SequenceInput(input_ids=ctx.batch.input_ids),
            shared=SequenceHeadShared(
                sequence=SequenceShared(position_ids=ctx.batch.position_ids),
                head=SequenceCausalLMHeadShared(labels=labels),
            ),
            state=PerplexityState(num_tokens=(labels != LM_IGNORE_INDEX).sum()),
        )

    def process_outputs(self, ctx: ProcessOutputsContext[SequenceCausalLMOutput, PerplexityState]):
        # The head returns the per-token cross-entropy loss. Padding positions hold zeros.
        self._loss_sum += ctx.pipeline_results.cross_entropy.sum()
        self._num_tokens += ctx.state.num_tokens

    def finalize(self, ctx: FinalizeContext) -> None:
        # Ranks that host no last pipeline stage add zeros, so a sum over the ranks with their own tokens covers
        # every batch once. Tensor-parallel ranks share their tokens, so the sum leaves them out.
        if self._dist_context.is_distributed:
            group = self._dist_context.mesh_for(METRIC_DOMAIN)[MetricDim.PP_DP_CP].get_group()
            dist.all_reduce(self._loss_sum, group=group)
            dist.all_reduce(self._num_tokens, group=group)

        mean_loss = (self._loss_sum / self._num_tokens).item()
        if self._dist_context.is_main_process:
            logger.info(f"Mean loss {mean_loss:.4f}, perplexity {math.exp(mean_loss):.2f}")


# --8<-- [end:task]


def main():
    config = EvaluationConfig.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")))
    # The checkpoint to evaluate, e.g. the original model or the export of a fine-tuning run.
    config.inference.model_stage_factory.source_checkpoint = Path(sys.argv[2])

    with init_dist_context(config.mesh) as dist_context:
        inference = InferenceConfigurator(
            dist_context=dist_context,
            parameters=config.inference,
            task_provider=lambda ctx: PerplexityTask(ctx.dist_context),
            model_provider=Qwen3Provider(config.model, lora=None),
            data_provider=AutoDataProvider(
                dataset_factory=lambda context: build_dataset(config.data, context.dist_context),
                collator=TextDataset.collate,
                config=config.auto_data,
            ),
        ).configure()

        inference.infer()


if __name__ == "__main__":
    main()
