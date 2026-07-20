from collections.abc import Mapping
from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from d9d.core.types import TensorSpec
from d9d.module.base import ModuleLateInit
from d9d.module.block.embedding import SplitTokenEmbeddings
from d9d.module.block.head import ClassificationHead, EmbeddingHead, SplitLanguageModellingHead
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode, create_hidden_states_aggregator
from d9d.module.block.normalization import RMSNorm
from d9d.module.block.positional import RotaryEmbeddingProvider, RotaryEmbeddingStyle
from d9d.module.model.io import (
    SequenceCausalLMOutput,
    SequenceCausalLMShared,
    SequenceClassificationOutput,
    SequenceEmbeddingOutput,
    SequenceInput,
    SequencePoolingShared,
    SequenceShared,
    SequenceTransfer,
)
from d9d.pipelining.api import (
    ModuleSupportsPipelining,
    PipelineStageInfo,
    StageBoundary,
    distribute_layers_for_pipeline_stage,
)

from .decoder_layer import Qwen3MoELayer
from .params import (
    Qwen3MoEForCausalLMParameters,
    Qwen3MoEForClassificationParameters,
    Qwen3MoEForEmbeddingParameters,
    Qwen3MoEParameters,
)


class Qwen3MoEModel(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
):
    """The Qwen3 Mixture-of-Experts (MoE) Transformer Decoder backbone.

    It is designed to be split across multiple pipeline stages.
    """

    def __init__(
        self,
        params: Qwen3MoEParameters,
        stage: PipelineStageInfo,
        hidden_states_snapshot_mode: HiddenStatesAggregationMode,
        enable_checkpointing: bool,
    ):
        """Constructs the Qwen3MoEModel object.

        Args:
            params: Configuration parameters for the full model.
            stage: Information about the pipeline stage this instance belongs to.
            hidden_states_snapshot_mode: Configures intermediate hidden state aggregation & snapshotting mode
            enable_checkpointing: If True, enables activation checkpointing for transformer layers to save memory.
        """
        super().__init__()

        if stage.is_current_stage_first:
            self.embed_tokens = SplitTokenEmbeddings(
                hidden_size=params.layer.hidden_size,
                split_vocab_size=params.split_vocab_size,
                split_order=params.split_vocab_order,
            )

        # we use ModuleDict here to properly handle pipelining and loading weights after the model
        # was pipelined
        layer_start, layer_end = distribute_layers_for_pipeline_stage(
            num_layers=params.num_hidden_layers,
            num_virtual_layers_pre=params.pipeline_num_virtual_layers_pre,  # embeddings
            num_virtual_layers_post=params.pipeline_num_virtual_layers_post,  # LM head
            stage=stage,
        )

        self._num_layers_before = layer_start
        self._layers_iter = list(map(str, range(layer_start, layer_end)))
        layers = nn.ModuleDict({str(layer_idx): Qwen3MoELayer(params=params.layer) for layer_idx in self._layers_iter})
        self.layers: Mapping[str, Qwen3MoELayer] = cast(Mapping[str, Qwen3MoELayer], layers)

        self.rope_provider = RotaryEmbeddingProvider(
            max_position_ids=params.max_position_ids,
            rope_base=params.rope_base,
            head_dim=params.layer.head_dim,
            style=RotaryEmbeddingStyle.HALF,
        )

        if stage.is_current_stage_last:
            self.norm = RMSNorm(params.layer.hidden_size, eps=params.layer.rms_norm_eps)

        self._stage = stage
        self._hidden_states_snapshot_mode = hidden_states_snapshot_mode
        self._hidden_size = params.layer.hidden_size
        self._enable_checkpointing = enable_checkpointing

    def output_dtype(self) -> torch.dtype:
        """Returns the data type of the model output hidden states.

        Returns:
            The output hidden states data type.
        """
        return self.layers[self._layers_iter[0]].input_layernorm.weight.dtype

    def forward(
        self,
        inputs: SequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequenceShared,
    ) -> SequenceTransfer[torch.Tensor]:
        """Executes the backbone forward pass for the current pipeline stage.

        Args:
            inputs: ``SequenceInput`` (token ids) on the first stage; the incoming
                ``SequenceTransfer`` otherwise.
            shared: The shared input broadcast to every stage (position ids and, if snapshotting is
                enabled, the aggregation mask).

        Returns:
            The produced ``SequenceTransfer`` (hidden states and, optionally, the updated snapshot).
        """
        state_aggregator = create_hidden_states_aggregator(
            self._hidden_states_snapshot_mode, shared.hidden_states_agg_mask
        )

        if self._stage.is_current_stage_first:
            first_inputs = cast(SequenceInput, inputs)
            last_hidden_states = self.embed_tokens(first_inputs.input_ids)
            hidden_states_snapshot = None
            state_aggregator.add_hidden_states(last_hidden_states)
        else:
            transfer_inputs = cast(SequenceTransfer[torch.Tensor], inputs)
            last_hidden_states = transfer_inputs.hidden_states
            hidden_states_snapshot = transfer_inputs.hidden_states_snapshot

        rope_params = self.rope_provider(shared.position_ids)

        for decoder_layer_name in self._layers_iter:
            decoder_layer = self.layers[decoder_layer_name]

            if self._enable_checkpointing:
                last_hidden_states = checkpoint(decoder_layer, last_hidden_states, rope_params, use_reentrant=False)
            else:
                last_hidden_states = decoder_layer(last_hidden_states, rope_params)

            state_aggregator.add_hidden_states(last_hidden_states)

        if self._stage.is_current_stage_last:
            last_hidden_states = self.norm(last_hidden_states)

        return SequenceTransfer(
            hidden_states=last_hidden_states,
            hidden_states_snapshot=state_aggregator.pack_with_snapshot(hidden_states_snapshot),
        )

    def reset_parameters(self):
        """Resets module parameters."""
        if self._stage.is_current_stage_first:
            self.embed_tokens.reset_parameters()

        self.rope_provider.reset_parameters()

        for decoder_layer_name in self._layers_iter:
            decoder_layer = self.layers[decoder_layer_name]
            decoder_layer.reset_parameters()

        if self._stage.is_current_stage_last:
            self.norm.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        """Describes the ``SequenceTransfer`` crossing the given boundary of this stage.

        Args:
            pipeline_input: A representative ``SequenceInput`` microbatch; only shapes are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        batch, seq = pipeline_input.input_ids.shape[0], pipeline_input.input_ids.shape[1]

        snapshot_spec = None
        if self._hidden_states_snapshot_mode != HiddenStatesAggregationMode.no:
            num_layers_before = self._num_layers_before + 1  # 1 for embedding
            if boundary is StageBoundary.outgoing:
                num_layers = num_layers_before + len(self.layers)
            else:
                num_layers = num_layers_before
            snapshot_spec = TensorSpec(shape=(num_layers, batch, self._hidden_size), dtype=self.output_dtype())

        return SequenceTransfer(
            hidden_states=TensorSpec(shape=(batch, seq, self._hidden_size), dtype=self.output_dtype()),
            hidden_states_snapshot=snapshot_spec,
        )


class Qwen3MoEForCausalLM(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceCausalLMShared, SequenceCausalLMOutput
    ],
):
    def __init__(
        self,
        params: Qwen3MoEForCausalLMParameters,
        stage: PipelineStageInfo,
        hidden_states_snapshot_mode: HiddenStatesAggregationMode,
        enable_checkpointing: bool,
    ):
        super().__init__()

        self.model = Qwen3MoEModel(
            params.model,
            stage,
            hidden_states_snapshot_mode=hidden_states_snapshot_mode,
            enable_checkpointing=enable_checkpointing,
        )

        if stage.is_current_stage_last:
            self.lm_head = SplitLanguageModellingHead(
                split_vocab_size=params.model.split_vocab_size,
                split_order=params.model.split_vocab_order,
                hidden_size=params.model.layer.hidden_size,
            )

        self._stage = stage
        self._hidden_size = params.model.layer.hidden_size

    def forward(
        self,
        inputs: SequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequenceCausalLMShared,
    ) -> SequenceTransfer[torch.Tensor] | SequenceCausalLMOutput:
        """Executes the model forward pass.

        If this is the last stage, it expects ``shared.labels`` to be provided and computes the
        cross-entropy loss (returned as per-token ``logps``).

        Args:
            inputs: ``SequenceInput`` on the first stage; incoming ``SequenceTransfer`` otherwise.
            shared: The shared input (position ids, aggregation mask, labels).

        Returns:
            The produced ``SequenceTransfer`` on non-last stages, or the ``CausalLMOutput`` on the
            last stage.
        """
        model_outputs = self.model(inputs, shared.sequence)
        if self._stage.is_current_stage_last:
            return SequenceCausalLMOutput(
                logps=self.lm_head(hidden_states=model_outputs.hidden_states, labels=shared.labels)
            )
        return model_outputs

    def reset_parameters(self):
        """Resets module parameters."""
        self.model.reset_parameters()

        if self._stage.is_current_stage_last:
            self.lm_head.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        return self.model.stage_transfer_spec(pipeline_input, boundary)


class Qwen3MoEForClassification(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequencePoolingShared, SequenceClassificationOutput
    ],
):
    """A Qwen3 MoE model wrapped with a Sequence/Token Classification head.

    It is designed to be split across multiple pipeline stages.
    """

    def __init__(
        self,
        params: Qwen3MoEForClassificationParameters,
        stage: PipelineStageInfo,
        hidden_states_snapshot_mode: HiddenStatesAggregationMode,
        enable_checkpointing: bool,
    ):
        """Constructs the Qwen3MoEForClassification object.

        Args:
            params: Full model configuration parameters.
            stage: Pipeline stage information for this instance.
            hidden_states_snapshot_mode: Configures intermediate hidden state aggregation & snapshotting mode.
            enable_checkpointing: Whether to enable activation checkpointing.
        """
        super().__init__()

        self.model = Qwen3MoEModel(
            params.model,
            stage,
            hidden_states_snapshot_mode=hidden_states_snapshot_mode,
            enable_checkpointing=enable_checkpointing,
        )

        if stage.is_current_stage_last:
            self.cls_head = ClassificationHead(
                hidden_size=params.model.layer.hidden_size,
                num_labels=params.num_labels,
                dropout=params.classifier_dropout,
            )

        self._stage = stage
        self._hidden_size = params.model.layer.hidden_size
        self._num_labels = params.num_labels

    def forward(
        self,
        inputs: SequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequencePoolingShared,
    ) -> SequenceTransfer[torch.Tensor] | SequenceClassificationOutput:
        """Executes the classification model forward pass.

        Args:
            inputs: ``SequenceInput`` on the first stage; incoming ``SequenceTransfer`` otherwise.
            shared: The shared input (position ids, aggregation mask, pooling mask).

        Returns:
            The produced ``SequenceTransfer`` on non-last stages, or the ``ClassificationOutput`` on
            the last stage.
        """
        model_outputs = self.model(inputs, shared.sequence)
        if self._stage.is_current_stage_last:
            return SequenceClassificationOutput(
                scores=self.cls_head(hidden_states=model_outputs.hidden_states, pooling_mask=shared.pooling_mask)
            )
        return model_outputs

    def reset_parameters(self):
        """Resets module parameters."""
        self.model.reset_parameters()

        if self._stage.is_current_stage_last:
            self.cls_head.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        return self.model.stage_transfer_spec(pipeline_input, boundary)


class Qwen3MoEForEmbedding(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequencePoolingShared, SequenceEmbeddingOutput
    ],
):
    """A Qwen3 MoE model wrapped with an Embedding head.

    It is designed to be split across multiple pipeline stages.
    """

    def __init__(
        self,
        params: Qwen3MoEForEmbeddingParameters,
        stage: PipelineStageInfo,
        hidden_states_snapshot_mode: HiddenStatesAggregationMode,
        enable_checkpointing: bool,
    ):
        """Constructs the Qwen3MoEForEmbedding object.

        Args:
            params: Full model configuration parameters.
            stage: Pipeline stage information for this instance.
            hidden_states_snapshot_mode: Configures intermediate hidden state aggregation & snapshotting mode.
            enable_checkpointing: Whether to enable activation checkpointing.
        """
        super().__init__()

        self.model = Qwen3MoEModel(
            params.model,
            stage,
            hidden_states_snapshot_mode=hidden_states_snapshot_mode,
            enable_checkpointing=enable_checkpointing,
        )

        if stage.is_current_stage_last:
            self.embedding_head = EmbeddingHead(
                hidden_size=params.model.layer.hidden_size,
                embedding_dim=params.embedding_dim,
                normalize=params.normalize,
            )

        self._stage = stage
        self._embedding_dim = (
            params.embedding_dim if params.embedding_dim is not None else params.model.layer.hidden_size
        )

    def forward(
        self,
        inputs: SequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequencePoolingShared,
    ) -> SequenceTransfer[torch.Tensor] | SequenceEmbeddingOutput:
        """Executes the embedding model forward pass.

        Args:
            inputs: ``SequenceInput`` on the first stage; incoming ``SequenceTransfer`` otherwise.
            shared: The shared input (position ids, aggregation mask, pooling mask).

        Returns:
            The produced ``SequenceTransfer`` on non-last stages, or the ``EmbeddingOutput`` on the
            last stage.
        """
        model_outputs = self.model(inputs, shared.sequence)
        if self._stage.is_current_stage_last:
            return SequenceEmbeddingOutput(
                embeddings=self.embedding_head(
                    hidden_states=model_outputs.hidden_states, pooling_mask=shared.pooling_mask
                )
            )
        return model_outputs

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.model.reset_parameters()

        if self._stage.is_current_stage_last:
            self.embedding_head.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        return self.model.stage_transfer_spec(pipeline_input, boundary)
