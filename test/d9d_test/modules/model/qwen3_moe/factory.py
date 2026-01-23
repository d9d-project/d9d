import torch
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model.qwen3_moe import (
    Qwen3MoEForCausalLM,
    Qwen3MoEForCausalLMParameters,
    Qwen3MoELayerParameters,
    Qwen3MoEParameters,
)
from d9d.pipelining.api import PipelineStageInfo


def build_decoder(stage: PipelineStageInfo, checkpointing: bool):
    torch.manual_seed(123)

    model = Qwen3MoEForCausalLM(
        params=Qwen3MoEForCausalLMParameters(
            model=Qwen3MoEParameters(
                layer=Qwen3MoELayerParameters(
                    hidden_size=512,
                    intermediate_size=256,
                    num_experts=8,
                    experts_top_k=7,
                    num_attention_heads=16,
                    num_key_value_heads=4,
                    rms_norm_eps=1e-5,
                    head_dim=32
                ),
                rope_base=10_000,
                max_position_ids=15000,
                num_hidden_layers=12,
                split_vocab_size={
                    "a": 40,
                    "b": 40,
                    "c": 20
                },
                split_vocab_order=["a", "b", "c"]
            )
        ),
        stage=stage,
        hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
        enable_checkpointing=checkpointing
    ).cuda().bfloat16()
    model.reset_parameters()
    return model


def build_decoder_inputs_hf():
    torch.manual_seed(428)

    input_ids = torch.randint(size=(8, 129), low=0, high=100 - 1, dtype=torch.long, device="cuda")

    labels = input_ids.clone()
    labels[0, :119] = -100
    labels[0, :100] = -100

    position_ids = torch.arange(0, 129, dtype=torch.long, device="cuda")[None, :].repeat(8, 1)

    return input_ids, position_ids, labels


def build_decoder_inputs_my():
    input_ids, position_ids, labels = build_decoder_inputs_hf()

    # shift
    input_ids = input_ids[:, :-1]
    position_ids = position_ids[:, :-1]
    labels = labels[:, 1:]

    return input_ids, position_ids, labels
