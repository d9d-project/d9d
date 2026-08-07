from collections.abc import Mapping
from typing import Generic, TypeVar, cast

import torch
from torch import nn

from d9d.core.types import TensorSpec
from d9d.module.base import ModuleLateInit
from d9d.module.block.head import (
    ClassificationHead,
    EmbeddingHead,
    SequenceCausalLMHeadShared,
    SequenceCausalLMOutput,
    SequenceClassificationOutput,
    SequenceEmbeddingOutput,
    SequencePoolingHeadShared,
    SplitLanguageModellingHead,
    TaskHead,
)
from d9d.module.model.backbone import DecoderBackbone
from d9d.module.model.head import (
    DEFAULT_HEAD_NAME_CAUSAL_LM,
    DEFAULT_HEAD_NAME_CLASSIFICATION,
    DEFAULT_HEAD_NAME_EMBEDDING,
    CausalLMHeadConfig,
    ClassificationHeadConfig,
    EmbeddingHeadConfig,
    build_decoder_head,
)
from d9d.module.model.io import (
    SequenceHeadsOutput,
    SequenceHeadsShared,
    SequenceInput,
    SequenceTransfer,
)
from d9d.pipelining.api import ModuleSupportsPipelining, PipelineStageInfo, StageBoundary

TBackbone = TypeVar("TBackbone", bound=DecoderBackbone)
THeadShared = TypeVar("THeadShared")
THeadOutput = TypeVar("THeadOutput")


class DecoderWithHeads(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput,
        SequenceTransfer[torch.Tensor],
        SequenceHeadsShared[THeadShared],
        SequenceHeadsOutput[THeadOutput],
    ],
    Generic[TBackbone, THeadShared, THeadOutput],
):
    """Composes one decoder backbone with a mapping of prebuilt named task heads.

    The backbone (``self.model``) and the heads (``self.heads``) are public, so a provider
    parallelizes and checkpoint-maps each independently. It does not build heads — that is the
    :func:`build_decoder_head` factory's job — so the class stays a pure composition of prebuilt
    modules and lets custom heads in on equal footing. Heads are attached only on the last pipeline
    stage.

    The head IO type parameters carry the composed heads' contract: a model with one kind of head
    names that head's shared input and output (see :class:`DecoderForCausalLM` and its siblings),
    while a model composed of several kinds names their unions.

    Type parameters:
        TBackbone: The backbone type, kept precise so ``parallelize_*`` accepts ``self.model``.
        THeadShared: The shared input accepted by the composed heads.
        THeadOutput: The output produced by the composed heads.
    """

    def __init__(
        self,
        backbone: TBackbone,
        heads: Mapping[str, TaskHead[THeadShared, THeadOutput]],
        stage: PipelineStageInfo,
    ):
        """Constructs the DecoderWithHeads object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            heads: A mapping of head names to prebuilt task heads, attached on the last stage
                as ``self.heads`` (FQN ``heads.<name>.*``).
            stage: Pipeline stage information for this instance.
        """
        super().__init__()

        self.model = backbone
        self._stage = stage

        if stage.is_current_stage_last:
            self.heads: Mapping[str, TaskHead[THeadShared, THeadOutput]] = cast(
                Mapping[str, TaskHead[THeadShared, THeadOutput]], nn.ModuleDict(dict(heads))
            )

    def forward(
        self,
        inputs: SequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequenceHeadsShared[THeadShared],
    ) -> SequenceTransfer[torch.Tensor] | SequenceHeadsOutput[THeadOutput]:
        """Executes the backbone and, on the last stage, every attached head.

        ``shared.sequence`` flows to the backbone; each head receives its own entry of
        ``shared.heads`` under the name it was composed with.

        Args:
            inputs: ``SequenceInput`` on the first stage; the incoming ``SequenceTransfer`` otherwise.
            shared: The backbone shared input plus the per-head shared inputs.

        Returns:
            The produced ``SequenceTransfer`` on non-last stages, or each head's output keyed by
                head name on the last stage.
        """
        model_outputs = self.model(inputs, shared.sequence)

        if not self._stage.is_current_stage_last:
            return model_outputs

        return {name: head(model_outputs.hidden_states, shared.heads[name]) for name, head in self.heads.items()}

    def reset_parameters(self) -> None:
        """Resets module parameters, delegating to the backbone and every head."""
        self.model.reset_parameters()

        if self._stage.is_current_stage_last:
            for head in self.heads.values():
                head.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        """Describes the ``SequenceTransfer`` crossing the given boundary, as the backbone does.

        Heads only run on the last stage, whose outgoing edge never transfers, so the transfer is
        entirely the backbone's.

        Args:
            pipeline_input: A representative ``SequenceInput`` microbatch; only shapes are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        return self.model.stage_transfer_spec(pipeline_input, boundary)


class DecoderForCausalLM(DecoderWithHeads[TBackbone, SequenceCausalLMHeadShared, SequenceCausalLMOutput]):
    """A decoder backbone composed with a single causal language modeling head.

    The single-head shorthand for the most common case: it builds the head itself and pins the
    composition IO to that head's shared input and output, so a task reads
    ``ctx.pipeline_results[...].logps`` without narrowing anything.
    """

    def __init__(
        self,
        backbone: TBackbone,
        stage: PipelineStageInfo,
        *,
        head_name: str = DEFAULT_HEAD_NAME_CAUSAL_LM,
    ):
        """Constructs the DecoderForCausalLM object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            stage: Pipeline stage information for this instance.
            head_name: The name the head is composed under (FQN ``heads.<head_name>.*``).
        """
        head = build_decoder_head(CausalLMHeadConfig(), backbone=backbone)
        super().__init__(backbone, {head_name: head}, stage)

        self._head_name = head_name

    @property
    def head(self) -> SplitLanguageModellingHead:
        """The composed language modeling head. Only attached on the last pipeline stage."""
        return cast(SplitLanguageModellingHead, self.heads[self._head_name])


class DecoderForClassification(DecoderWithHeads[TBackbone, SequencePoolingHeadShared, SequenceClassificationOutput]):
    """A decoder backbone composed with a single classification head.

    The single-head shorthand for the most common case: it builds the head itself and pins the
    composition IO to that head's shared input and output, so a task reads
    ``ctx.pipeline_results[...].scores`` without narrowing anything.
    """

    def __init__(
        self,
        backbone: TBackbone,
        config: ClassificationHeadConfig,
        stage: PipelineStageInfo,
        *,
        head_name: str = DEFAULT_HEAD_NAME_CLASSIFICATION,
    ):
        """Constructs the DecoderForClassification object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            config: Configuration of the classification head to build.
            stage: Pipeline stage information for this instance.
            head_name: The name the head is composed under (FQN ``heads.<head_name>.*``).
        """
        head = build_decoder_head(config, backbone=backbone)
        super().__init__(backbone, {head_name: head}, stage)

        self._head_name = head_name

    @property
    def head(self) -> ClassificationHead:
        """The composed classification head. Only attached on the last pipeline stage."""
        return cast(ClassificationHead, self.heads[self._head_name])


class DecoderForEmbedding(DecoderWithHeads[TBackbone, SequencePoolingHeadShared, SequenceEmbeddingOutput]):
    """A decoder backbone composed with a single embedding head.

    The single-head shorthand for the most common case: it builds the head itself and pins the
    composition IO to that head's shared input and output, so a task reads
    ``ctx.pipeline_results[...].embeddings`` without narrowing anything.
    """

    def __init__(
        self,
        backbone: TBackbone,
        config: EmbeddingHeadConfig,
        stage: PipelineStageInfo,
        *,
        head_name: str = DEFAULT_HEAD_NAME_EMBEDDING,
    ):
        """Constructs the DecoderForEmbedding object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            config: Configuration of the embedding head to build.
            stage: Pipeline stage information for this instance.
            head_name: The name the head is composed under (FQN ``heads.<head_name>.*``).
        """
        head = build_decoder_head(config, backbone=backbone)
        super().__init__(backbone, {head_name: head}, stage)

        self._head_name = head_name

    @property
    def head(self) -> EmbeddingHead:
        """The composed embedding head. Only attached on the last pipeline stage."""
        return cast(EmbeddingHead, self.heads[self._head_name])
