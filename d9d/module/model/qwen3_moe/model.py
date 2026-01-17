import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from d9d.module.base import ModuleLateInit
from d9d.module.block.embedding import SplitTokenEmbeddings
from d9d.module.block.head import SplitLanguageModellingHead
from d9d.module.block.positional import RotaryEmbeddingProvider
from d9d.module.model.qwen3_moe import Qwen3MoEParameters, Qwen3MoELayer, Qwen3MoEForCausalLMParameters
from d9d.pipelining.api import ModuleSupportsPipelining, PipelineStageInfo, distribute_layers_for_pipeline_stage


def _aggregate_hidden_states(hidden_states: torch.Tensor, agg_mask: torch.Tensor):
    orig_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    num_tokens = agg_mask.sum(dim=1)[:, None]
    masked_states = hidden_states * agg_mask[:, :, None]
    averaged_states = masked_states.sum(dim=1) / num_tokens
    return averaged_states.to(orig_dtype)


class Qwen3MoEModel(nn.Module, ModuleLateInit, ModuleSupportsPipelining):
    """
    The Qwen3 Mixture-of-Experts (MoE) Transformer Decoder backbone.

    It is designed to be split across multiple pipeline stages.
    """

    def __init__(
            self,
            params: Qwen3MoEParameters,
            stage: PipelineStageInfo,
            output_hidden_states_snapshot: bool,
            enable_checkpointing: bool
    ):
        """
        Constructs the Qwen3MoEModel object.

        Args:
            params: Configuration parameters for the full model.
            stage: Information about the pipeline stage this instance belongs to.
            output_hidden_states_snapshot: If True, intermediate hidden states pooled by mask will be collected
                and returned.
            enable_checkpointing: If True, enables activation checkpointing for transformer layers to save memory.
        """

        super().__init__()

        if stage.is_current_stage_first:
            self.embed_tokens = SplitTokenEmbeddings(
                hidden_size=params.layer.hidden_size,
                split_vocab_size=params.split_vocab_size,
                split_order=params.split_vocab_order
            )

        # we use ModuleDict here to properly handle pipelining and loading weights after the model
        # was pipelined
        layer_start, layer_end = distribute_layers_for_pipeline_stage(
            num_layers=params.num_hidden_layers,
            num_virtual_layers_pre=0,  # embeddings
            num_virtual_layers_post=2,  # LM head
            stage=stage
        )

        self._num_layers_before = layer_start
        self._layers_iter = list(map(str, range(layer_start, layer_end)))
        self.layers = nn.ModuleDict({
            str(layer_idx): Qwen3MoELayer(params=params.layer) for layer_idx in self._layers_iter
        })

        self.rope_provider = RotaryEmbeddingProvider(
            max_position_ids=params.max_position_ids,
            rope_base=params.rope_base,
            head_dim=params.layer.head_dim
        )

        if stage.is_current_stage_last:
            self.norm = nn.RMSNorm(
                normalized_shape=params.layer.hidden_size,
                eps=params.layer.rms_norm_eps
            )

        self._stage = stage
        self._output_hidden_states_snapshot = output_hidden_states_snapshot
        self._hidden_size = params.layer.hidden_size
        self._enable_checkpointing = enable_checkpointing

    def output_dtype(self) -> torch.dtype:
        """
        Returns the data type of the model output hidden states.
        """
        return self.layers[self._layers_iter[0]].input_layernorm.weight.dtype

    def forward(
            self,
            input_ids: torch.Tensor | None = None,
            hidden_states: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            hidden_states_snapshot: torch.Tensor | None = None,
            hidden_states_agg_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Executes the forward pass for the current pipeline stage.

        Args:
            input_ids: Indices of input sequence tokens. Required if this is the
                first pipeline stage.
            hidden_states: Hidden states from the previous pipeline stage. Required
                if this is not the first pipeline stage.
            position_ids: Indices of positions of each input sequence tokens in the
                position embeddings.
            hidden_states_snapshot: Accumulated tensor of aggregated hidden states
                from previous stages. Used if snapshotting is enabled.
            hidden_states_agg_mask: Mask used to aggregate hidden states for
                snapshots.

        Returns:
            A dictionary containing:
            *   'hidden_states': The output of the last layer in this stage.
            *   'hidden_states_snapshot': (Optional) The updated snapshot tensor.
        """
        hidden_states_to_output = []

        if input_ids is not None:
            last_hidden_states = self.embed_tokens(input_ids)
            if self._output_hidden_states_snapshot:
                hidden_states_to_output.append(_aggregate_hidden_states(last_hidden_states, hidden_states_agg_mask))
        else:
            last_hidden_states = hidden_states

        rope_params = self.rope_provider(position_ids)

        for decoder_layer_name in self._layers_iter:
            decoder_layer = self.layers[decoder_layer_name]

            if self._enable_checkpointing:
                last_hidden_states = checkpoint(
                    decoder_layer, last_hidden_states, rope_params,
                    use_reentrant=False
                )
            else:
                last_hidden_states = decoder_layer(last_hidden_states, rope_params)

            if self._output_hidden_states_snapshot:
                hidden_states_to_output.append(_aggregate_hidden_states(last_hidden_states, hidden_states_agg_mask))

        if self._stage.is_current_stage_last:
            last_hidden_states = self.norm(last_hidden_states)


        outputs = {
            'hidden_states': last_hidden_states
        }

        if self._output_hidden_states_snapshot:
            hidden_states_to_output = torch.stack(hidden_states_to_output, dim=0)
            if hidden_states_snapshot is not None:
                hidden_states_to_output = torch.cat([hidden_states_snapshot, hidden_states_to_output], dim=0)
            outputs['hidden_states_snapshot'] = hidden_states_to_output

        return outputs

    def reset_moe_stats(self):
        """
        Resets routing statistics for all MoE layers in this stage.
        """

        for layer_name in self._layers_iter:
            self.layers[layer_name].reset_moe_stats()

    @property
    def moe_tokens_per_expert(self):
        """
        Retrieves the number of tokens routed to each expert across all layers.

        Returns:
            A tensor of shape (num_local_layers, num_experts) containing counts.
        """

        return torch.stack(
            [self.layers[layer_name].moe_tokens_per_expert for layer_name in self._layers_iter],
            dim=0
        )

    def reset_parameters(self):
        """Resets module parameters"""

        if self._stage.is_current_stage_first:
            self.embed_tokens.reset_parameters()

        self.rope_provider.reset_parameters()

        for decoder_layer_name in self._layers_iter:
            decoder_layer = self.layers[decoder_layer_name]
            decoder_layer.reset_parameters()

        if self._stage.is_current_stage_last:
            self.norm.reset_parameters()

    def infer_stage_inputs_from_pipeline_inputs(
            self, inputs: dict[str, torch.Tensor], n_microbatches: int
    ) -> dict[str, torch.Tensor]:
        input_ids = inputs['input_ids']

        pp_inputs = {}

        # for calculation - input ids or prev hidden state
        if self._stage.is_current_stage_first:
            pp_inputs['input_ids'] = torch.empty(
                (input_ids.shape[0] // n_microbatches, input_ids.shape[1]),
                dtype=torch.long,
                device=input_ids.device
            )
        else:
            pp_inputs['hidden_states'] = torch.empty(
                (input_ids.shape[0] // n_microbatches, input_ids.shape[1], self._hidden_size),
                dtype=self.output_dtype(),
                device=input_ids.device
            )
            if self._output_hidden_states_snapshot:
                num_layers_before = self._num_layers_before + 1  # 1 for embedding
                pp_inputs['hidden_states_snapshot'] = torch.empty(
                    (num_layers_before, input_ids.shape[0] // n_microbatches, self._hidden_size),
                    dtype=self.output_dtype(),
                    device=input_ids.device
                )

        return pp_inputs

    def infer_stage_outputs_from_pipeline_inputs(
            self, inputs: dict[str, torch.Tensor], n_microbatches: int
    ) -> dict[str, torch.Tensor]:
        input_ids = inputs['input_ids']

        # for calculation - last hidden state
        pp_outputs = {
            'hidden_states': torch.empty(
                (input_ids.shape[0] // n_microbatches, input_ids.shape[1], self._hidden_size),
                dtype=self.output_dtype(),
                device=input_ids.device
            )
        }

        # for state caching
        if self._output_hidden_states_snapshot:
            num_layers_before = self._num_layers_before + 1
            num_layers_current = len(self.layers)
            num_layers_after = num_layers_before + num_layers_current
            pp_outputs['hidden_states_snapshot'] = torch.empty(
                (num_layers_after, input_ids.shape[0] // n_microbatches, self._hidden_size),
                dtype=self.output_dtype(),
                device=input_ids.device
            )

        return pp_outputs


class Qwen3MoEForCausalLM(nn.Module, ModuleLateInit, ModuleSupportsPipelining):
    """
    A Qwen3 MoE model wrapped with a Causal Language Modeling head.

    It is designed to be split across multiple pipeline stages.
    """

    def __init__(
            self,
            params: Qwen3MoEForCausalLMParameters,
            stage: PipelineStageInfo,
            output_hidden_states_snapshot: bool,
            enable_checkpointing: bool
    ):
        """
        Constructs the Qwen3MoEForCausalLM object.

        Args:
            params: Full model configuration parameters.
            stage: Pipeline stage information for this instance.
            output_hidden_states_snapshot: Whether to capture aggregated hidden states.
            enable_checkpointing: Whether to enable activation checkpointing.
        """

        super().__init__()

        self.model = Qwen3MoEModel(
            params,
            stage,
            output_hidden_states_snapshot=output_hidden_states_snapshot,
            enable_checkpointing=enable_checkpointing
        )

        if stage.is_current_stage_last:
            self.lm_head = SplitLanguageModellingHead(
                split_vocab_size=params.split_vocab_size,
                split_order=params.split_vocab_order,
                hidden_size=params.hidden_size
            )

        self._stage = stage
        self._hidden_size = params.hidden_size

    def forward(
            self,
            input_ids: torch.Tensor | None = None,
            hidden_states: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            hidden_states_snapshot: torch.Tensor | None = None,
            hidden_states_agg_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Executes the model forward pass.

        If this is the last stage, it expects `labels` to be provided and computes
        the cross-entropy loss (returned as 'logps' typically representing per-token loss).

        Args:
            input_ids: Input token IDS (for Stage 0).
            hidden_states: Hidden states from previous stage (for Stage > 0).
            position_ids: Positional indices for RoPE.
            hidden_states_snapshot: Intermediate state collector.
            hidden_states_agg_mask: Mask for state aggregation.
            labels: Target tokens for loss computation (Last Stage).

        Returns:
            Dictionary containing 'hidden_states', optionally 'hidden_states_snapshot',
            and per-token 'logps' if on the last stage.
        """

        model_outputs = self.model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            position_ids=position_ids,
            hidden_states_snapshot=hidden_states_snapshot,
            hidden_states_agg_mask=hidden_states_agg_mask
        )
        if self._stage.is_current_stage_last:
            lm_out = self.lm_head(
                hidden_states=model_outputs['hidden_states'],
                labels=labels
            )
            model_outputs['logps'] = lm_out
        return model_outputs

    def reset_parameters(self):
        """
        Resets module parameters.
        """

        self.model.reset_parameters()

        if self._stage.is_current_stage_last:
            self.lm_head.reset_parameters()

    def reset_moe_stats(self):
        """
        Resets MoE routing statistics in the backbone.
        """

        self.model.reset_moe_stats()

    @property
    def moe_tokens_per_expert(self):
        """
        Accesses MoE routing statistics from the backbone.
        """

        return self.model.moe_tokens_per_expert

    def infer_stage_inputs_from_pipeline_inputs(
            self, inputs: dict[str, torch.Tensor], n_microbatches: int
    ) -> dict[str, torch.Tensor]:
        return self.model.infer_stage_inputs_from_pipeline_inputs(inputs, n_microbatches)

    def infer_stage_outputs_from_pipeline_inputs(
            self, inputs: dict[str, torch.Tensor], n_microbatches: int
    ) -> dict[str, torch.Tensor]:
        pp_outputs = self.model.infer_stage_outputs_from_pipeline_inputs(inputs, n_microbatches)

        if self._stage.is_current_stage_last:
            pp_outputs['logps'] = torch.empty(inputs['input_ids'].shape, dtype=torch.float32)

        return pp_outputs
