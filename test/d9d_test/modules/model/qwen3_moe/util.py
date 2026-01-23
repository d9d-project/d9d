from d9d.module.model.qwen3_moe import Qwen3MoEForCausalLM, Qwen3MoELayer, Qwen3MoEModel
from d9d.pipelining.api import PipelineStageInfo
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeForCausalLM, Qwen3MoeModel

from d9d_test.modules.block.attention.util import (
    check_grouped_query_attention_qwen3_moe_grad,
    clone_grouped_query_attention_qwen3_moe,
)
from d9d_test.modules.block.embedding.util import check_embeddings_grad, clone_embeddings
from d9d_test.modules.block.head.util import check_lm_head_grad, clone_lm_head
from d9d_test.modules.block.moe.util import check_moe_qwen3_moe_grad, clone_moe_weights_qwen3_moe
from d9d_test.modules.checkers import check_grad_distance


def clone_layer_weights(my: Qwen3MoELayer, hf: Qwen3MoeDecoderLayer):
    clone_grouped_query_attention_qwen3_moe(my.self_attn, hf.self_attn)
    clone_moe_weights_qwen3_moe(my.mlp, hf.mlp)
    my.post_attention_layernorm.weight.data = hf.post_attention_layernorm.weight.data.detach().clone()
    my.input_layernorm.weight.data = hf.input_layernorm.weight.data.detach().clone()


def clone_model_weights(my: Qwen3MoEModel, hf: Qwen3MoeModel, stage: PipelineStageInfo):
    if stage.is_current_stage_first:
        clone_embeddings(my.embed_tokens, hf.embed_tokens)

    for layer_i, layer_my in my.layers.items():
        layer_i_int = int(layer_i)
        layer_hf = hf.layers[layer_i_int]
        clone_layer_weights(layer_my, layer_hf)

    if stage.is_current_stage_last:
        my.norm.weight.data = hf.norm.weight.data.detach().clone()


def clone_lm_model_weights(my: Qwen3MoEForCausalLM, hf: Qwen3MoeForCausalLM, stage: PipelineStageInfo):
    clone_model_weights(my.model, hf.model, stage=stage)
    if stage.is_current_stage_last:
        clone_lm_head(my.lm_head, hf.lm_head)


def check_layer_grad(my: Qwen3MoELayer, hf: Qwen3MoeDecoderLayer, is_dist: bool):
    check_grouped_query_attention_qwen3_moe_grad(my.self_attn, hf.self_attn, is_dist=is_dist)
    check_moe_qwen3_moe_grad(my.mlp, hf.mlp, is_dist=is_dist)
    check_grad_distance(
        my.post_attention_layernorm.weight.grad[None, :],
        hf.post_attention_layernorm.weight.grad[None, :],
        is_dist=is_dist
    )
    check_grad_distance(
        my.input_layernorm.weight.grad[None, :],
        hf.input_layernorm.weight.grad[None, :],
        is_dist=is_dist
    )


def check_model_grad(my: Qwen3MoEModel, hf: Qwen3MoeModel, is_dist: bool, stage: PipelineStageInfo):
    if stage.is_current_stage_first:
        check_embeddings_grad(my.embed_tokens, hf.embed_tokens, is_dist=is_dist)

    for layer_i, layer_my in my.layers.items():
        layer_i_int = int(layer_i)
        layer_hf = hf.layers[layer_i_int]
        check_layer_grad(layer_my, layer_hf, is_dist=is_dist)

    if stage.is_current_stage_last:
        check_grad_distance(
            my.norm.weight.grad[None, :],
            hf.norm.weight.grad[None, :],
            is_dist=is_dist
        )


def check_lm_model_grad(my: Qwen3MoEForCausalLM, hf: Qwen3MoeForCausalLM, is_dist: bool, stage: PipelineStageInfo):
    if stage.is_current_stage_last:
        check_lm_head_grad(my.lm_head, hf.lm_head, is_dist=is_dist)

    check_model_grad(my.model, hf.model, is_dist=is_dist, stage=stage)
