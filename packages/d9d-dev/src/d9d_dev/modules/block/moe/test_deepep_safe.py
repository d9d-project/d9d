import sys

import pytest
from d9d.module.block.moe.layer import MoELayer


@pytest.mark.local
def test_deepep_not_imported_on_init():
    MoELayer(
        hidden_dim=128,
        intermediate_dim_grouped=256,
        num_grouped_experts=4,
        top_k=2,
        router_renormalize_probabilities=True,
    )

    assert "deep_ep_cpp" not in sys.modules
