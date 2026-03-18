from d9d.module.model.qwen3_dense import (
    Qwen3DenseForCausalLM,
    Qwen3DenseForClassification,
    Qwen3DenseLayer,
    Qwen3DenseModel,
)
from d9d.pipelining.api import PipelineStageInfo
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3ForSequenceClassification,
    Qwen3Model,
)

from d9d_test.modules.block.attention.util import (
    check_grouped_query_attention_qwen3_dense_grad,
    clone_grouped_query_attention_qwen3_dense,
)
from d9d_test.modules.block.dense.util import check_dense_qwen3_dense_grad, clone_dense_weights_qwen3_dense
from d9d_test.modules.block.embedding.util import check_embeddings_grad, clone_embeddings
from d9d_test.modules.block.head.util import (
    check_classification_head_grad,
    check_lm_head_grad,
    clone_classification_head,
    clone_lm_head,
)
from d9d_test.modules.checkers import check_grad_distance


def clone_layer_weights(my: Qwen3DenseLayer, hf: Qwen3DecoderLayer):
    clone_grouped_query_attention_qwen3_dense(my.self_attn, hf.self_attn)
    clone_dense_weights_qwen3_dense(my.mlp, hf.mlp)
    my.post_attention_layernorm.weight.data = hf.post_attention_layernorm.weight.data.detach().clone()
    my.input_layernorm.weight.data = hf.input_layernorm.weight.data.detach().clone()


def clone_model_weights(my: Qwen3DenseModel, hf: Qwen3Model, stage: PipelineStageInfo):
    if stage.is_current_stage_first:
        clone_embeddings(my.embed_tokens, hf.embed_tokens)

    for layer_i, layer_my in my.layers.items():
        layer_i_int = int(layer_i)
        layer_hf = hf.layers[layer_i_int]
        clone_layer_weights(layer_my, layer_hf)

    if stage.is_current_stage_last:
        my.norm.weight.data = hf.norm.weight.data.detach().clone()


def clone_lm_model_weights(my: Qwen3DenseForCausalLM, hf: Qwen3ForCausalLM, stage: PipelineStageInfo):
    clone_model_weights(my.model, hf.model, stage=stage)
    if stage.is_current_stage_last:
        clone_lm_head(my.lm_head, hf.lm_head)


def clone_cls_model_weights(
    my: Qwen3DenseForClassification, hf: Qwen3ForSequenceClassification, stage: PipelineStageInfo
):
    clone_model_weights(my.model, hf.model, stage=stage)
    if stage.is_current_stage_last:
        clone_classification_head(my.cls_head, hf.score)


def check_layer_grad(my: Qwen3DenseLayer, hf: Qwen3DecoderLayer):
    check_grouped_query_attention_qwen3_dense_grad(my.self_attn, hf.self_attn)
    check_dense_qwen3_dense_grad(my.mlp, hf.mlp)
    check_grad_distance(
        my.post_attention_layernorm.weight.grad[None, :], hf.post_attention_layernorm.weight.grad[None, :]
    )
    check_grad_distance(my.input_layernorm.weight.grad[None, :], hf.input_layernorm.weight.grad[None, :])


def check_model_grad(my: Qwen3DenseModel, hf: Qwen3Model, stage: PipelineStageInfo):
    if stage.is_current_stage_first:
        check_embeddings_grad(my.embed_tokens, hf.embed_tokens)

    for layer_i, layer_my in my.layers.items():
        layer_i_int = int(layer_i)
        layer_hf = hf.layers[layer_i_int]
        check_layer_grad(layer_my, layer_hf)

    if stage.is_current_stage_last:
        check_grad_distance(my.norm.weight.grad[None, :], hf.norm.weight.grad[None, :])


def check_lm_model_grad(my: Qwen3DenseForCausalLM, hf: Qwen3ForCausalLM, stage: PipelineStageInfo):
    if stage.is_current_stage_last:
        check_lm_head_grad(my.lm_head, hf.lm_head)

    check_model_grad(my.model, hf.model, stage=stage)


def check_cls_model_grad(my: Qwen3DenseForClassification, hf: Qwen3ForSequenceClassification, stage: PipelineStageInfo):
    if stage.is_current_stage_last:
        check_classification_head_grad(my.cls_head, hf.score)

    check_model_grad(my.model, hf.model, stage=stage)
