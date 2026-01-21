import dataclasses
import os
import random

import numpy as np
import pytest
import torch
from d9d.core.dist_context.device_mesh_domains import FLAT_DOMAIN
from d9d.core.dist_ops import all_gather_object
from d9d.internals.determinism import set_seeds


@dataclasses.dataclass(frozen=True, kw_only=True)
class RandomState:
    torch: float
    numpy: float
    random: float
    hash_seed: str


def _get_random_states() -> RandomState:
    return RandomState(
        torch=torch.rand(1).item(),
        numpy=np.random.rand(),
        random=random.random(),
        hash_seed=os.environ["PYTHONHASHSEED"]
    )


def check_states_same(*states: RandomState):
    assert len(set(states)) == 1


def check_states_different(*states: RandomState):
    torch_values = {x.torch for x in states}
    numpy_values = {x.numpy for x in states}
    random_values = {x.random for x in states}
    hash_seed_values = {x.hash_seed for x in states}

    assert len(torch_values) == len(states)
    assert len(numpy_values) == len(states)
    assert len(random_values) == len(states)
    assert len(hash_seed_values) == len(states)


@pytest.mark.local
def test_set_seeds_local_determinism(dist_ctx_local):
    # 1. Set seed A
    set_seeds(dist_ctx_local, seed=42)
    state_1 = _get_random_states()

    # 2. Set seed B
    set_seeds(dist_ctx_local, seed=999)
    state_2 = _get_random_states()

    # Different seeds produce different results
    check_states_different(state_1, state_2)

    # 3. Set seed A again
    set_seeds(dist_ctx_local, seed=42)
    state_3 = _get_random_states()

    check_states_same(state_1, state_3)


@pytest.mark.distributed
def test_seed_variation_over_pp(dist_ctx_pp_dpr):
    world_size = dist_ctx_pp_dpr.mesh_for(FLAT_DOMAIN).size()
    pp_size = dist_ctx_pp_dpr.mesh_params.pipeline_parallel
    non_pp_size = world_size // dist_ctx_pp_dpr.mesh_params.pipeline_parallel

    set_seeds(dist_ctx_pp_dpr, seed=42)

    # Generate a random number on each rank
    local_state = _get_random_states()

    all_states = all_gather_object(local_state, group=dist_ctx_pp_dpr.mesh_for(FLAT_DOMAIN).get_group())
    all_states_by_pp = [all_states[pp_rank * non_pp_size: (pp_rank + 1) * non_pp_size] for pp_rank in range(pp_size)]

    for pp_batch in all_states_by_pp:
        #  all states within PP are the same
        check_states_same(*pp_batch)

    # all states across PP are different
    check_states_different(*(x[0] for x in all_states_by_pp))
