import torch
from d9d.module.block.moe import MoELayer
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from d9d_test.modules.checkers import check_grad_distance


def clone_moe_weights_qwen3_moe(my: MoELayer, hf: Qwen3MoeSparseMoeBlock):
    my.router.gate.weight.data = hf.gate.weight.data.detach().clone()
    my.grouped_experts.gate_proj.weight.data = (
        torch.stack([exp.gate_proj.weight.data.T for exp in hf.experts], dim=0).detach().clone()
    )
    my.grouped_experts.up_proj.weight.data = (
        torch.stack([exp.up_proj.weight.data.T for exp in hf.experts], dim=0).detach().clone()
    )
    my.grouped_experts.down_proj.weight.data = (
        torch.stack([exp.down_proj.weight.data.T for exp in hf.experts], dim=0).detach().clone()
    )


def _optional_expert(weight: nn.Parameter):
    if weight.grad is None:
        return torch.zeros_like(weight).T
    else:
        return weight.grad.T


def check_moe_qwen3_moe_grad(my: MoELayer, hf: Qwen3MoeSparseMoeBlock):
    for my_grad, hf_grad in [
        (my.router.gate.weight.grad, hf.gate.weight.grad),
        (
            my.grouped_experts.gate_proj.weight.grad,
            torch.stack([_optional_expert(x.gate_proj.weight) for x in hf.experts], dim=0),
        ),
        (
            my.grouped_experts.up_proj.weight.grad,
            torch.stack([_optional_expert(x.up_proj.weight) for x in hf.experts], dim=0),
        ),
        (
            my.grouped_experts.down_proj.weight.grad,
            torch.stack([_optional_expert(x.down_proj.weight) for x in hf.experts], dim=0),
        ),
    ]:
        check_grad_distance(my_grad.flatten(start_dim=-2), hf_grad.flatten(start_dim=-2))
