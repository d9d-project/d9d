from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import nn

from d9d.nn import ModuleLateInit
from d9d.nn.embedding import SplitTokenEmbeddings
from d9d.nn.hidden_states_aggregator import HiddenStatesAggregationMode, create_hidden_states_aggregator
from d9d.nn.normalization import RMSNorm
from d9d.nn.positional import RotaryEmbeddingProvider, RotaryEmbeddingStyle
from d9d.pipelining.api import (
    ModuleSupportsPipelining,
    PipelineStageInfo,
    StageBoundary,
    StageSegment,
    TensorSpec,
    distribute_layers_for_pipeline_stage,
    run_segments,
)

from ..io import SequenceInput, SequenceShared, SequenceTransfer
from ..segments import SequenceSegmentConstants, build_sequence_segments
from .decoder_layer import Qwen3DenseLayer
from .params import Qwen3DenseParameters


class Qwen3DenseModel(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
):
    """The Qwen3 Dense transformer decoder backbone.

    The backbone supports pipeline parallelism: each instance holds the layers of one stage.
    """

    def __init__(
        self,
        params: Qwen3DenseParameters,
        stage: PipelineStageInfo,
        hidden_states_snapshot_mode: HiddenStatesAggregationMode,
        enable_checkpointing: bool,
    ):
        """Constructs the ``Qwen3DenseModel`` object.

        Args:
            params: Configuration parameters for the full model.
            stage: The pipeline stage this instance belongs to.
            hidden_states_snapshot_mode: How to aggregate intermediate hidden states into snapshots.
            enable_checkpointing: Whether to apply activation checkpointing to each transformer layer.
        """
        super().__init__()

        if stage.is_current_stage_first:
            self.embed_tokens = SplitTokenEmbeddings(
                hidden_size=params.layer.hidden_size,
                split_vocab_size=params.split_vocab_size,
                split_order=params.split_vocab_order,
            )

        # A ModuleDict keyed by global layer index keeps the parameter FQNs the same on every
        # pipeline stage, so checkpoints load regardless of the stage split.
        layer_start, layer_end = distribute_layers_for_pipeline_stage(
            num_layers=params.num_hidden_layers,
            num_virtual_layers_pre=params.pipeline_num_virtual_layers_pre,  # Embedding
            num_virtual_layers_post=params.pipeline_num_virtual_layers_post,  # LM head
            stage=stage,
        )

        self._num_layers_before = layer_start
        self._layers_iter = list(map(str, range(layer_start, layer_end)))
        layers = nn.ModuleDict(
            {str(layer_idx): Qwen3DenseLayer(params=params.layer) for layer_idx in self._layers_iter}
        )
        self.layers: Mapping[str, Qwen3DenseLayer] = cast(Mapping[str, Qwen3DenseLayer], layers)

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
        self._enable_checkpointing = enable_checkpointing

        # Task heads read these dimensions from the backbone.
        self._hidden_size = params.layer.hidden_size
        self._split_vocab_size = params.split_vocab_size
        self._split_vocab_order = params.split_vocab_order

    @property
    def hidden_size(self) -> int:
        """The size of the backbone hidden states."""
        return self._hidden_size

    @property
    def split_vocab_size(self) -> Mapping[str, int]:
        """The mapping of vocabulary segment names to their sizes."""
        return self._split_vocab_size

    @property
    def split_vocab_order(self) -> Sequence[str]:
        """The order in which vocabulary segments are concatenated."""
        return self._split_vocab_order

    def output_dtype(self) -> torch.dtype:
        """Returns the dtype of the output hidden states.

        Returns:
            The dtype of the output hidden states.
        """
        return self.layers[self._layers_iter[0]].input_layernorm.weight.dtype

    def forward(
        self,
        inputs: SequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequenceShared,
    ) -> SequenceTransfer[torch.Tensor]:
        """Executes the backbone forward pass for the current pipeline stage.

        Args:
            inputs: ``SequenceInput`` with token ids on the first stage, or the incoming
                ``SequenceTransfer`` on other stages.
            shared: The backbone shared input: position ids and, if snapshotting is enabled, the
                aggregation mask.

        Returns:
            The produced ``SequenceTransfer`` with the hidden states and, if enabled, the updated
                snapshot.
        """
        return run_segments(self, inputs, shared)

    def prepare_segment_constants(self, shared: SequenceShared) -> SequenceSegmentConstants:
        """Builds the constants of one microbatch.

        Args:
            shared: The backbone shared input.

        Returns:
            The RoPE embeddings and the snapshot aggregator.
        """
        return SequenceSegmentConstants(
            position_embeddings=self.rope_provider(shared.position_ids),
            aggregator=create_hidden_states_aggregator(
                self._hidden_states_snapshot_mode, shared.hidden_states_agg_mask
            ),
        )

    def stage_segments(self) -> Sequence[StageSegment]:
        """Returns the segments of this stage, in forward order."""
        return build_sequence_segments(
            embed_tokens=self.embed_tokens if self._stage.is_current_stage_first else None,
            layers=[(name, self.layers[name]) for name in self._layers_iter],
            norm=self.norm if self._stage.is_current_stage_last else None,
            enable_checkpointing=self._enable_checkpointing,
        )

    def reset_parameters(self):
        """Resets the module parameters."""
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
            pipeline_input: A representative ``SequenceInput`` microbatch. Only its shapes are read.
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
            snapshot_spec = TensorSpec(shape=(num_layers, batch, self.hidden_size), dtype=self.output_dtype())

        return SequenceTransfer(
            hidden_states=TensorSpec(shape=(batch, seq, self.hidden_size), dtype=self.output_dtype()),
            hidden_states_snapshot=snapshot_spec,
        )
