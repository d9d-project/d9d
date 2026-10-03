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
    CausalLMHeadConfig,
    ClassificationHeadConfig,
    EmbeddingHeadConfig,
    build_decoder_head,
)
from d9d.module.model.io import (
    SequenceHeadShared,
    SequenceHeadsOutput,
    SequenceHeadsShared,
    SequenceInput,
    SequenceTransfer,
)
from d9d.pipelining.api import ModuleSupportsPipelining, PipelineStageInfo, StageBoundary

TBackbone = TypeVar("TBackbone", bound=DecoderBackbone)
THead = TypeVar("THead", bound=TaskHead)
THeadShared = TypeVar("THeadShared")
THeadOutput = TypeVar("THeadOutput")

SINGLE_HEAD_PREFIX = "head."
"""FQN prefix of the task head in a single-head decoder (any ``DecoderWithHead``)."""


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

    For a model with a single head, use ``DecoderWithHead`` instead. With one head there is nothing
    to key, and this class would make every caller name the only head.

    The backbone (``self.model``) and the heads (``self.heads``) are public, so a provider can
    parallelize and checkpoint-map each of them separately. This class does not build heads: use
    ``build_decoder_head`` for built-in heads, or pass custom heads in the same way. Heads are
    attached only on the last pipeline stage.

    For heads of one kind, the head I/O type parameters are that kind's shared input and output.
    For heads of several kinds, they are the unions of those types.

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
        """Constructs the ``DecoderWithHeads`` object.

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
        """Runs the backbone and, on the last stage, every attached head.

        The backbone receives ``shared.sequence``. Each head receives the entry of ``shared.heads``
        stored under its name.

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
        """Resets the parameters of the backbone and of every head."""
        self.model.reset_parameters()

        if self._stage.is_current_stage_last:
            for head in self.heads.values():
                head.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        """Describes the ``SequenceTransfer`` crossing the given boundary, as the backbone does.

        Heads only run on the last stage, which has no outgoing transfer, so the backbone alone
        defines the transfer.

        Args:
            pipeline_input: A representative ``SequenceInput`` microbatch. Only its shapes are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        return self.model.stage_transfer_spec(pipeline_input, boundary)


class DecoderWithHead(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput,
        SequenceTransfer[torch.Tensor],
        SequenceHeadShared[THeadShared],
        THeadOutput,
    ],
    Generic[TBackbone, THead, THeadShared, THeadOutput],
):
    """Composes one decoder backbone with exactly one prebuilt task head.

    This is the single-head counterpart of ``DecoderWithHeads``, not a special case of it. With one
    head there is nothing to key. The head is ``self.head`` (FQN ``head.*``), its shared input is a
    field, and the model returns the head's output instead of a one-entry mapping. For example, a
    causal LM task reads ``ctx.pipeline_results.logps`` directly.

    The backbone (``self.model``) and the head (``self.head``) are public, so a provider can
    parallelize and checkpoint-map each of them separately. The head is attached only on the last
    pipeline stage.

    Type parameters:
        TBackbone: The backbone type, kept precise so ``parallelize_*`` accepts ``self.model``.
        THead: The head type, kept precise so ``parallelize_*_head`` accepts ``self.head``.
        THeadShared: The shared input accepted by the composed head.
        THeadOutput: The output produced by the composed head.
    """

    def __init__(self, backbone: TBackbone, head: THead, stage: PipelineStageInfo):
        """Constructs the ``DecoderWithHead`` object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            head: The prebuilt task head, attached on the last stage as ``self.head``
                (FQN ``head.*``).
            stage: Pipeline stage information for this instance.
        """
        super().__init__()

        self.model = backbone
        self._stage = stage

        if stage.is_current_stage_last:
            self.head: THead = head

    def forward(
        self,
        inputs: SequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequenceHeadShared[THeadShared],
    ) -> SequenceTransfer[torch.Tensor] | THeadOutput:
        """Runs the backbone and, on the last stage, the attached head.

        Args:
            inputs: ``SequenceInput`` on the first stage; the incoming ``SequenceTransfer`` otherwise.
            shared: The backbone shared input plus the head's shared input.

        Returns:
            The produced ``SequenceTransfer`` on non-last stages, or the head's output on the last
                stage.
        """
        model_outputs = self.model(inputs, shared.sequence)

        if not self._stage.is_current_stage_last:
            return model_outputs

        return self.head(model_outputs.hidden_states, shared.head)

    def reset_parameters(self) -> None:
        """Resets the parameters of the backbone and of the head."""
        self.model.reset_parameters()

        if self._stage.is_current_stage_last:
            self.head.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        """Describes the ``SequenceTransfer`` crossing the given boundary, as the backbone does.

        The head only runs on the last stage, which has no outgoing transfer, so the backbone alone
        defines the transfer.

        Args:
            pipeline_input: A representative ``SequenceInput`` microbatch. Only its shapes are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        return self.model.stage_transfer_spec(pipeline_input, boundary)


class DecoderForCausalLM(
    DecoderWithHead[TBackbone, SplitLanguageModellingHead, SequenceCausalLMHeadShared, SequenceCausalLMOutput]
):
    """A decoder backbone composed with a single causal language modeling head.

    It builds the head itself, so composing the model is one call. Its I/O types are fixed to that
    head's shared input and output.
    """

    def __init__(self, backbone: TBackbone, stage: PipelineStageInfo):
        """Constructs the ``DecoderForCausalLM`` object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            stage: Pipeline stage information for this instance.
        """
        head = cast(SplitLanguageModellingHead, build_decoder_head(CausalLMHeadConfig(), backbone=backbone))
        super().__init__(backbone, head, stage)


class DecoderForClassification(
    DecoderWithHead[TBackbone, ClassificationHead, SequencePoolingHeadShared, SequenceClassificationOutput]
):
    """A decoder backbone composed with a single classification head.

    It builds the head itself, so composing the model is one call. Its I/O types are fixed to that
    head's shared input and output.
    """

    def __init__(self, backbone: TBackbone, config: ClassificationHeadConfig, stage: PipelineStageInfo):
        """Constructs the ``DecoderForClassification`` object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            config: Configuration of the classification head to build.
            stage: Pipeline stage information for this instance.
        """
        head = cast(ClassificationHead, build_decoder_head(config, backbone=backbone))
        super().__init__(backbone, head, stage)


class DecoderForEmbedding(
    DecoderWithHead[TBackbone, EmbeddingHead, SequencePoolingHeadShared, SequenceEmbeddingOutput]
):
    """A decoder backbone composed with a single embedding head.

    It builds the head itself, so composing the model is one call. Its I/O types are fixed to that
    head's shared input and output.
    """

    def __init__(self, backbone: TBackbone, config: EmbeddingHeadConfig, stage: PipelineStageInfo):
        """Constructs the ``DecoderForEmbedding`` object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            config: Configuration of the embedding head to build.
            stage: Pipeline stage information for this instance.
        """
        head = cast(EmbeddingHead, build_decoder_head(config, backbone=backbone))
        super().__init__(backbone, head, stage)
