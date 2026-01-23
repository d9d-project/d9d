import torch
from d9d.module.block.head import SplitLanguageModellingHead
from torch import nn

from d9d_test.modules.checkers import check_grad_distance


def clone_lm_head(my: SplitLanguageModellingHead, hf: nn.Linear):
    size_per_head = {
        name: head.out_features
        for name, head
        in my.lm_head.items()
    }
    prepend_size = 0
    for part_name in my._split_order:
        part_size = size_per_head[part_name]
        hf_part = hf.weight.data[prepend_size:prepend_size + part_size].detach().clone()
        my.lm_head[part_name].weight.data = hf_part
        prepend_size += part_size


def check_lm_head_grad(my: SplitLanguageModellingHead, hf: nn.Linear, is_dist: bool):
    my_grad = torch.cat(
        [my.lm_head[part_name].weight.grad for part_name in my._split_order],
        dim=0
    )
    hf_grad = hf.weight.grad

    check_grad_distance(my_grad, hf_grad, is_dist=is_dist)
