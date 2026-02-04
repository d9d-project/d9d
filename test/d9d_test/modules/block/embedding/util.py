import torch
from d9d.module.block.embedding import SplitTokenEmbeddings
from torch import nn

from d9d_test.modules.checkers import check_grad_distance


def clone_embeddings(my: SplitTokenEmbeddings, hf: nn.Embedding):
    size_per_part = {
        name: part.num_embeddings
        for name, part
        in my.token_embedding.items()
    }
    prepend_size = 0
    for part_name in my._split_order:
        part_size = size_per_part[part_name]
        hf_part = hf.weight.data[prepend_size:prepend_size + part_size].detach().clone()
        my.token_embedding[part_name].weight.data = hf_part
        prepend_size += part_size


def check_embeddings_grad(my: SplitTokenEmbeddings, hf: nn.Embedding):
    my_grad = torch.cat(
        [my.token_embedding[part_name].weight.grad for part_name in my._split_order],
        dim=0
    )

    hf_grad = hf.weight.grad
    check_grad_distance(my_grad, hf_grad)
