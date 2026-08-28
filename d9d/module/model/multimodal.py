from collections.abc import Mapping, Sequence
from typing import Generic, TypeVar, cast

import torch
from torch import nn

from d9d.core.types import TensorSpec
from d9d.module.base import ModalityEncoder, ModuleLateInit
from d9d.module.block.embedding import merge_media_embeddings
from d9d.module.model.backbone import DecoderBackbone
from d9d.module.model.io import (
    MultimodalSequenceInput,
    SequenceInput,
    SequenceShared,
    SequenceTransfer,
)
from d9d.pipelining.api import ModuleSupportsPipelining, PipelineStageInfo, StageBoundary

TBackbone = TypeVar("TBackbone", bound=DecoderBackbone[SequenceInput])


class MultimodalBackbone(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        MultimodalSequenceInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
    Generic[TBackbone],
):
    """Wraps a text-only decoder backbone with a modality encoder, and is itself a backbone.

    This is the composition that makes a model multimodal: on the first stage it embeds the tokens,
    encodes the packed media stream, merges the media embeddings into the placeholder positions,
    and hands the merged embeddings to the wrapped backbone via ``SequenceInput.inputs_embeds``.
    Every later stage is delegated unchanged — media never crosses a stage boundary.

    Because it satisfies :class:`DecoderBackbone` itself (with ``MultimodalSequenceInput`` as its
    pipeline input), it composes with the task heads exactly as a text-only backbone does::

        model = DecoderForCausalLM(MultimodalBackbone(backbone, encoder, stage), stage)

    The wrapped backbone (``self.model``) and the encoder (``self.encoder``) are public, so a
    provider parallelizes and checkpoint-maps each independently. The encoder lives strictly on the
    first stage; its cost is accounted for with ``pipeline_num_virtual_layers_pre``.

    The encoder runs on *every* first-stage microbatch, including media-free ones (which carry a
    dummy segment built by ``pad_empty_media``). This is required: all microbatches of a pack must
    share one input PyTree structure, and skipping the encoder on some ranks would desynchronize
    FSDP's lazily-triggered all-gathers. ``merge_media_embeddings`` attaches a zero-valued
    contribution in that case, so gradients stay structurally identical across ranks.

    Type parameters:
        TBackbone: The wrapped backbone type, kept precise so ``parallelize_*`` accepts
            ``self.model``.
    """

    def __init__(self, backbone: TBackbone, encoder: ModalityEncoder, stage: PipelineStageInfo):
        """Constructs the MultimodalBackbone object.

        Args:
            backbone: The text-only decoder backbone to wrap, exposed as ``self.model``
                (FQN ``model.*``).
            encoder: The modality encoder consuming the packed media stream, attached on the first
                stage as ``self.encoder`` (FQN ``encoder.*``).
            stage: Pipeline stage information for this instance.
        """
        super().__init__()

        self.model = backbone
        self._stage = stage

        if stage.is_current_stage_first:
            self.encoder = encoder

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the backbone hidden states, as reported by the wrapped backbone."""
        return self.model.hidden_size

    @property
    def split_vocab_size(self) -> Mapping[str, int]:
        """Mapping of vocabulary segment names to their sizes, from the wrapped backbone."""
        return self.model.split_vocab_size

    @property
    def split_vocab_order(self) -> Sequence[str]:
        """The order in which vocabulary segments are concatenated, from the wrapped backbone."""
        return self.model.split_vocab_order

    def forward(
        self,
        inputs: MultimodalSequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequenceShared,
    ) -> SequenceTransfer[torch.Tensor]:
        """Encodes and merges media on the first stage, then delegates to the wrapped backbone.

        Args:
            inputs: ``MultimodalSequenceInput`` on the first stage; the incoming
                ``SequenceTransfer`` otherwise.
            shared: The backbone shared input broadcast to every stage. For a multimodal model
                ``position_ids`` carries the three MRoPE planes, shape ``(3, batch, seq)``.

        Returns:
            The produced ``SequenceTransfer``.
        """
        if not self._stage.is_current_stage_first:
            # Off the first stage the incoming payload is always the inter-stage transfer, which
            # the wrapped backbone consumes unchanged.
            return self.model(cast(SequenceTransfer[torch.Tensor], inputs), shared)

        first_inputs = cast(MultimodalSequenceInput, inputs)
        token_embeddings = self.model.embed_tokens(first_inputs.input_ids)
        media_embeddings = self.encoder(first_inputs.media)
        merged = merge_media_embeddings(token_embeddings, first_inputs.media_token_mask, media_embeddings)

        return self.model(SequenceInput(input_ids=first_inputs.input_ids, inputs_embeds=merged), shared)

    def reset_parameters(self) -> None:
        """Resets module parameters, delegating to the wrapped backbone and the encoder."""
        self.model.reset_parameters()

        if self._stage.is_current_stage_first:
            self.encoder.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: MultimodalSequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        """Describes the ``SequenceTransfer`` crossing the given boundary, as the backbone does.

        Media never crosses a stage boundary, so the transfer is entirely the wrapped backbone's.
        The token ids are forwarded because that is all the backbone reads shapes from.

        Args:
            pipeline_input: A representative ``MultimodalSequenceInput`` microbatch; only shapes
                are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        return self.model.stage_transfer_spec(SequenceInput(input_ids=pipeline_input.input_ids), boundary)
