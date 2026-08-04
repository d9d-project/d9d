from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from d9d.core.types import TensorSpec
from d9d.module.base import ModuleLateInit
from d9d.module.block.embedding import SplitTokenEmbeddings
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode, create_hidden_states_aggregator
from d9d.module.block.normalization import RMSNorm
from d9d.module.block.positional import MultimodalRotaryEmbeddingProvider
from d9d.module.model.io import (
    SequenceInput,
    SequenceShared,
    SequenceTransfer,
)
from d9d.pipelining.api import (
    ModuleSupportsPipelining,
    PipelineStageInfo,
    StageBoundary,
    distribute_layers_for_pipeline_stage,
)

from .decoder_layer import Qwen3p5MoEFullAttentionLayer, Qwen3p5MoELinearAttentionLayer
from .params import (
    Qwen3p5MoEParameters,
)

Qwen3p5MoEDecoderLayer = Qwen3p5MoEFullAttentionLayer | Qwen3p5MoELinearAttentionLayer


def _expand_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Expands text position ids to the three MRoPE position planes when needed.

    Args:
        position_ids: Position ids of shape ``(batch, seq)`` (text-only) or ``(3, batch, seq)``.

    Returns:
        Position ids of shape ``(3, batch, seq)``.
    """
    if position_ids.ndim == 2:
        return position_ids.unsqueeze(0).expand(3, -1, -1)
    return position_ids


class Qwen3p5MoEModel(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
):
    """The Qwen3.5 Mixture-of-Experts (MoE) Transformer Decoder backbone.

    Interleaves linear-attention (Gated DeltaNet) and full-attention decoder layers according to
    ``full_attention_interval``. It is designed to be split across multiple pipeline stages.
    """

    def __init__(
        self,
        params: Qwen3p5MoEParameters,
        stage: PipelineStageInfo,
        hidden_states_snapshot_mode: HiddenStatesAggregationMode,
        enable_checkpointing: bool,
    ):
        """Constructs the Qwen3p5MoEModel object.

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
            num_virtual_layers_pre=params.pipeline_num_virtual_layers_pre,  # embeddings & vision encoder
            num_virtual_layers_post=params.pipeline_num_virtual_layers_post,  # LM head
            stage=stage,
        )

        self._num_layers_before = layer_start
        self._layers_iter = list(map(str, range(layer_start, layer_end)))
        layers = nn.ModuleDict(
            {
                str(layer_idx): (
                    Qwen3p5MoEFullAttentionLayer(params=params.layer)
                    if params.is_full_attention_layer(layer_idx)
                    else Qwen3p5MoELinearAttentionLayer(params=params.layer)
                )
                for layer_idx in range(layer_start, layer_end)
            }
        )
        self.layers: Mapping[str, Qwen3p5MoEDecoderLayer] = cast(Mapping[str, Qwen3p5MoEDecoderLayer], layers)

        self.rope_provider = MultimodalRotaryEmbeddingProvider(
            rope_base=params.rope_base,
            rope_dim=params.layer.rope_dim,
            max_position_ids=params.max_position_ids,
            mrope_section=params.mrope_section,
            interleaved=True,
        )

        if stage.is_current_stage_last:
            self.norm = RMSNorm(params.layer.hidden_size, eps=params.layer.rms_norm_eps, zero_centered=True)

        self._stage = stage
        self._hidden_states_snapshot_mode = hidden_states_snapshot_mode
        self._enable_checkpointing = enable_checkpointing

        # backbone-shared dimensions the composed task heads derive from
        self._hidden_size = params.layer.hidden_size
        self._split_vocab_size = params.split_vocab_size
        self._split_vocab_order = params.split_vocab_order

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the backbone hidden states."""
        return self._hidden_size

    @property
    def split_vocab_size(self) -> Mapping[str, int]:
        """Mapping of vocabulary segment names to their sizes."""
        return self._split_vocab_size

    @property
    def split_vocab_order(self) -> Sequence[str]:
        """The order in which vocabulary segments are concatenated."""
        return self._split_vocab_order

    def output_dtype(self) -> torch.dtype:
        """Returns the data type of the model output hidden states.

        Returns:
            The output hidden states data type.
        """
        return self.layers[self._layers_iter[0]].input_layernorm.weight.dtype

    def forward_embedded(
        self,
        hidden_states: torch.Tensor,
        hidden_states_snapshot: torch.Tensor | None,
        shared: SequenceShared,
    ) -> SequenceTransfer[torch.Tensor]:
        """Runs this stage's decoder layers over already-embedded hidden states.

        Args:
            hidden_states: The input hidden states, shape ``(batch, seq, hidden)``.
            hidden_states_snapshot: The accumulated snapshot carried from previous stages, if any.
            shared: The shared input broadcast to every stage.

        Returns:
            The produced ``SequenceTransfer``.
        """
        state_aggregator = create_hidden_states_aggregator(
            self._hidden_states_snapshot_mode, shared.hidden_states_agg_mask
        )

        if self._stage.is_current_stage_first:
            state_aggregator.add_hidden_states(hidden_states)

        rope_params = self.rope_provider(_expand_position_ids(shared.position_ids))

        for decoder_layer_name in self._layers_iter:
            decoder_layer = self.layers[decoder_layer_name]

            if self._enable_checkpointing:
                hidden_states = checkpoint(decoder_layer, hidden_states, rope_params, use_reentrant=False)
            else:
                hidden_states = decoder_layer(hidden_states, rope_params)

            state_aggregator.add_hidden_states(hidden_states)

        if self._stage.is_current_stage_last:
            hidden_states = self.norm(hidden_states)

        return SequenceTransfer(
            hidden_states=hidden_states,
            hidden_states_snapshot=state_aggregator.pack_with_snapshot(hidden_states_snapshot),
        )

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
        if self._stage.is_current_stage_first:
            first_inputs = cast(SequenceInput, inputs)
            if first_inputs.inputs_embeds is not None:
                hidden_states = first_inputs.inputs_embeds
            else:
                hidden_states = self.embed_tokens(first_inputs.input_ids)
            hidden_states_snapshot = None
        else:
            transfer_inputs = cast(SequenceTransfer[torch.Tensor], inputs)
            hidden_states = transfer_inputs.hidden_states
            hidden_states_snapshot = transfer_inputs.hidden_states_snapshot

        return self.forward_embedded(hidden_states, hidden_states_snapshot, shared)

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
