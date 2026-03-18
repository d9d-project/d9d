from d9d.module.block.ffn import SwiGLU
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from d9d_test.modules.checkers import check_grad_distance


def clone_dense_weights_qwen3_dense(my: SwiGLU, hf: Qwen3MLP):
    my.gate_proj.weight.data = hf.gate_proj.weight.data.detach().clone()
    my.up_proj.weight.data = hf.up_proj.weight.data.detach().clone()
    my.down_proj.weight.data = hf.down_proj.weight.data.detach().clone()


def check_dense_qwen3_dense_grad(my: SwiGLU, hf: Qwen3MLP):
    pairs = (
        ("gate_proj.weight", my.gate_proj.weight.grad, hf.gate_proj.weight.grad),
        ("up_proj.weight", my.up_proj.weight.grad, hf.up_proj.weight.grad),
        ("down_proj.weight", my.down_proj.weight.grad, hf.down_proj.weight.grad),
    )

    for _, my_grad, hf_grad in pairs:
        assert my_grad is not None
        assert hf_grad is not None
        check_grad_distance(my_grad, hf_grad)
