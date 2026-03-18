import torch
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model.qwen3_dense import (
    Qwen3DenseForCausalLM,
    Qwen3DenseForCausalLMParameters,
    Qwen3DenseLayerParameters,
    Qwen3DenseParameters,
)
from d9d.pipelining.api import PipelineStageInfo


def build_decoder(stage: PipelineStageInfo, checkpointing: bool):
    torch.manual_seed(123)

    model = (
        Qwen3DenseForCausalLM(
            params=Qwen3DenseForCausalLMParameters(
                model=Qwen3DenseParameters(
                    layer=Qwen3DenseLayerParameters(
                        hidden_size=512,
                        intermediate_size=256,
                        num_attention_heads=16,
                        num_key_value_heads=4,
                        rms_norm_eps=1e-5,
                        head_dim=32,
                    ),
                    rope_base=10_000,
                    max_position_ids=15000,
                    num_hidden_layers=12,
                    split_vocab_size={"a": 40, "b": 40, "c": 20},
                    split_vocab_order=["a", "b", "c"],
                )
            ),
            stage=stage,
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=checkpointing,
        )
        .cuda()
        .bfloat16()
    )
    model.reset_parameters()
    return model
