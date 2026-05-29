import sys

import pytest
from d9d.module.block.moe.layer import MoELayer
from d9d.module.block.moe.router import TopKRouter


@pytest.mark.local
def test_deepep_not_imported_on_init():
    MoELayer(
        hidden_dim=128,
        intermediate_dim_grouped=256,
        num_grouped_experts=4,
        router=TopKRouter(dim=128, num_experts=4, top_k=2, renormalize_probabilities=True),
    )

    assert "deep_ep_cpp" not in sys.modules
