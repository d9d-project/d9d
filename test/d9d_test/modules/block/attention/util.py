from d9d.module.block.attention import GroupedQueryAttention, MultiHeadLatentAttention
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from d9d_test.modules.checkers import check_grad_distance


def clone_grouped_query_attention_qwen3_moe(my: GroupedQueryAttention, hf: Qwen3MoeAttention):
    my.q_proj.weight.data = hf.q_proj.weight.data.detach().clone()
    my.k_proj.weight.data = hf.k_proj.weight.data.detach().clone()
    my.v_proj.weight.data = hf.v_proj.weight.data.detach().clone()
    my.o_proj.weight.data = hf.o_proj.weight.data.detach().clone()
    my.q_norm.weight.data = hf.q_norm.weight.data.detach().clone()
    my.k_norm.weight.data = hf.k_norm.weight.data.detach().clone()


def check_grouped_query_attention_qwen3_moe_grad(my: GroupedQueryAttention, hf: Qwen3MoeAttention):
    for my_grad, hf_grad in [
        (my.q_proj.weight.grad, hf.q_proj.weight.grad),
        (my.k_proj.weight.grad, hf.k_proj.weight.grad),
        (my.v_proj.weight.grad, hf.v_proj.weight.grad),
        (my.o_proj.weight.grad, hf.o_proj.weight.grad),
    ]:
        check_grad_distance(my_grad, hf_grad)

    for my_grad, hf_grad in [
        (my.q_norm.weight.grad, hf.q_norm.weight.grad),
        (my.k_norm.weight.grad, hf.k_norm.weight.grad),
    ]:
        check_grad_distance(my_grad[None, :], hf_grad[None, :])


def clone_grouped_query_attention_qwen3_dense(my: GroupedQueryAttention, hf: Qwen3Attention):
    my.q_proj.weight.data = hf.q_proj.weight.data.detach().clone()
    my.k_proj.weight.data = hf.k_proj.weight.data.detach().clone()
    my.v_proj.weight.data = hf.v_proj.weight.data.detach().clone()
    my.o_proj.weight.data = hf.o_proj.weight.data.detach().clone()
    my.q_norm.weight.data = hf.q_norm.weight.data.detach().clone()
    my.k_norm.weight.data = hf.k_norm.weight.data.detach().clone()


def check_grouped_query_attention_qwen3_dense_grad(my: GroupedQueryAttention, hf: Qwen3Attention):
    for my_grad, hf_grad in [
        (my.q_proj.weight.grad, hf.q_proj.weight.grad),
        (my.k_proj.weight.grad, hf.k_proj.weight.grad),
        (my.v_proj.weight.grad, hf.v_proj.weight.grad),
        (my.o_proj.weight.grad, hf.o_proj.weight.grad),
    ]:
        check_grad_distance(my_grad, hf_grad)

    for my_grad, hf_grad in [
        (my.q_norm.weight.grad, hf.q_norm.weight.grad),
        (my.k_norm.weight.grad, hf.k_norm.weight.grad),
    ]:
        check_grad_distance(my_grad[None, :], hf_grad[None, :])


def clone_mla_deepseek_v2(my: MultiHeadLatentAttention, hf: DeepseekV2Attention) -> None:
    """Copy weights from HF DeepseekV2Attention to d9d MultiHeadLatentAttention.

    HF uses ``a``/``b`` naming (e.g. ``q_a_proj``), d9d uses ``down``/``up``
    (e.g. ``q_proj.down_proj``). This function maps between the two conventions.
    """
    if my.q_lora_rank is not None:
        my.q_proj.down_proj.weight.data = hf.q_a_proj.weight.data.detach().clone()
        my.q_proj.norm.weight.data = hf.q_a_layernorm.weight.data.detach().clone()
        my.q_proj.up_proj.weight.data = hf.q_b_proj.weight.data.detach().clone()
    else:
        my.q_proj.weight.data = hf.q_proj.weight.data.detach().clone()
    my.kv_down_proj.weight.data = hf.kv_a_proj_with_mqa.weight.data.detach().clone()
    my.kv_down_norm.weight.data = hf.kv_a_layernorm.weight.data.detach().clone()
    my.kv_up_proj.weight.data = hf.kv_b_proj.weight.data.detach().clone()
    my.o_proj.weight.data = hf.o_proj.weight.data.detach().clone()


def check_mla_deepseek_v2_grad(my: MultiHeadLatentAttention, hf: DeepseekV2Attention) -> None:
    """Validate gradient similarity between d9d MLA and HF DeepseekV2Attention."""
    proj_pairs: list[tuple[object, object]] = [
        (my.kv_down_proj.weight.grad, hf.kv_a_proj_with_mqa.weight.grad),
        (my.kv_up_proj.weight.grad, hf.kv_b_proj.weight.grad),
        (my.o_proj.weight.grad, hf.o_proj.weight.grad),
    ]
    if my.q_lora_rank is not None:
        proj_pairs += [
            (my.q_proj.down_proj.weight.grad, hf.q_a_proj.weight.grad),
            (my.q_proj.up_proj.weight.grad, hf.q_b_proj.weight.grad),
        ]
    else:
        proj_pairs += [(my.q_proj.weight.grad, hf.q_proj.weight.grad)]

    for my_grad, hf_grad in proj_pairs:
        check_grad_distance(my_grad, hf_grad)

    # Layernorm grads are 1-D — add batch dim for check_grad_distance
    check_grad_distance(
        my.kv_down_norm.weight.grad[None, :],
        hf.kv_a_layernorm.weight.grad[None, :],
    )
    if my.q_lora_rank is not None:
        check_grad_distance(
            my.q_proj.norm.weight.grad[None, :],
            hf.q_a_layernorm.weight.grad[None, :],
        )
