import copy
from contextlib import nullcontext

import pytest
import torch
from d9d.dist_context import DATA_SHARD_DIMS, DenseDomainConfig, DeviceMeshParameters
from d9d.parallel.runtime import ParameterRuntime
from d9d.pipelining.stage.windows import WindowChain, backward_chains, forward_chains
from torch.distributed.tensor import distribute_tensor

from d9d_dev.nn.helper.tolerances import forward_tolerance_for
from d9d_dev.parallel.runtime_step import place_hsdp
from d9d_dev.pipelining.window_models import _Revisited


@pytest.mark.distributed
@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.parametrize("split", [False, True])
def test_repeated_groups_match_global_gradients_after_rebinding(dist_ctx_factory, replicas, split):
    context = dist_ctx_factory(
        DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8 // replicas))
    )
    model = _Revisited().cuda().double()
    reference = copy.deepcopy(model)
    place_hsdp(model, context)
    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=32)
    runtime.register_stage(0, model)

    def inputs(rank, microbatch):
        return torch.full((2, 3), (rank + microbatch + 1) / 16, device="cuda", dtype=torch.float64)

    for _ in range(2):
        runtime.bind()
        try:
            for _ in range(2):
                runtime.zero_grad()
                reference.zero_grad(set_to_none=True)
                chains = [WindowChain(model, 0, runtime) for _ in range(2)]
                for microbatch, chain in enumerate(chains):
                    chain.start_forward(inputs(torch.distributed.get_rank(), microbatch), None)
                outputs = forward_chains(chains, runtime, 0, [nullcontext(), nullcontext()])
                for chain, output in zip(chains, outputs, strict=True):
                    chain.start_backward([output.square().sum()], None)
                if split:
                    for microbatch, chain in enumerate(chains):
                        chain.backward_input()
                        chain.backward_weight(microbatch == len(chains) - 1)
                else:
                    backward_chains(chains, runtime, 0, True, [nullcontext(), nullcontext()])
                runtime.wait()
                _check_global_gradients(model, reference, inputs)
        finally:
            runtime.unbind()


def _check_global_gradients(model, reference, inputs):
    for rank in range(8):
        for microbatch in range(2):
            reference.reference(inputs(rank, microbatch)).square().sum().backward()
    for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
        shard = distribute_tensor(expected.grad, parameter.device_mesh, parameter.placements).to_local()
        tolerance = forward_tolerance_for(shard.dtype)
        torch.testing.assert_close(parameter.grad.to_local(), shard, rtol=tolerance.rtol, atol=tolerance.atol)
