from d9d.module.block.attention import GroupedQueryAttention
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
        (my.o_proj.weight.grad, hf.o_proj.weight.grad)
    ]:
        check_grad_distance(my_grad, hf_grad)

    for my_grad, hf_grad in [
        (my.q_norm.weight.grad, hf.q_norm.weight.grad),
        (my.k_norm.weight.grad, hf.k_norm.weight.grad),
    ]:
        check_grad_distance(my_grad[None, :], hf_grad[None, :])
