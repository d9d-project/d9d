import zlib
from collections.abc import Callable
from typing import Any

import torch
from d9d.dist_context import DATA_SHARD_DIMS, DenseDomainConfig, DeviceMeshParameters
from d9d.models.io import SequenceHeadShared, SequenceInput, SequenceShared
from d9d.models.qwen3_dense import Qwen3DenseLayerParameters, Qwen3DenseParameters
from d9d.nn.head import SequenceCausalLMHeadShared
from d9d.parallel.runtime import ParameterRuntime
from torch import nn
from torch.distributed.tensor import DTensor, Shard

MESHES = [
    DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)),
    DeviceMeshParameters(data_parallel=8),
    DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4)),
]

# The small Qwen3 dense model of the runtime tests.
PARAMS = Qwen3DenseParameters(
    layer=Qwen3DenseLayerParameters(
        hidden_size=64,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        head_dim=16,
    ),
    num_hidden_layers=8,
    rope_base=10000,
    max_position_ids=128,
    split_vocab_size={"a": 50},
    split_vocab_order=["a"],
)
_BATCH, _SEQ = 2, 8


def global_weight(name: str, param: torch.Tensor) -> torch.Tensor:
    """Returns the full weight of a parameter, the same on every rank and in every build of the model."""
    generator = torch.Generator(device=param.device).manual_seed(zlib.crc32(name.encode()))
    weight = torch.randn(param.shape, device=param.device, generator=generator)
    return (0.02 * weight + 0.1).to(param.dtype)


def batches(replica: int) -> tuple[list[SequenceInput], list[SequenceHeadShared]]:
    """Returns four microbatches of tokens for the vocabulary of ``PARAMS``, the same for one replica on every rank."""
    inputs, shared = [], []
    for microbatch in range(4):
        generator = torch.Generator(device="cuda").manual_seed(100 * replica + microbatch)
        ids = torch.randint(0, 50, (_BATCH, _SEQ), device="cuda", generator=generator)
        positions = torch.arange(_SEQ, device="cuda").expand(_BATCH, _SEQ)
        inputs.append(SequenceInput(input_ids=ids))
        shared.append(
            SequenceHeadShared(
                sequence=SequenceShared(position_ids=positions), head=SequenceCausalLMHeadShared(labels=ids)
            )
        )
    return inputs, shared


def local(param: DTensor) -> torch.Tensor:
    """Returns the local shard of a distributed parameter, which writes through to the parameter."""
    return param.data.to_local()


def shard_of(full: torch.Tensor, param: DTensor) -> torch.Tensor:
    """Returns the part of a full tensor that the local shard of a distributed parameter holds."""
    for dim, placement in enumerate(param.placements):
        if isinstance(placement, Shard):
            mesh = param.device_mesh
            sub_mesh = mesh[mesh.mesh_dim_names[dim]]
            rows = -(-full.shape[placement.dim] // sub_mesh.size())
            return full.narrow(placement.dim, rows * sub_mesh.get_local_rank(), local(param).shape[placement.dim])
    return full


def make_runtime(bucket_size_mib: int = 32, **kwargs: Any) -> ParameterRuntime:
    """Builds a runtime that gathers along the data shard dimensions of d9d meshes."""
    return ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=bucket_size_mib, **kwargs)


def step_with_runtime(module: nn.Module, closure: Callable[[], None]) -> ParameterRuntime:
    """Runs a forward and backward pass of a module as one region and reduces its gradients.

    The returned runtime stays bound, so the sharded parameters keep their gradients until ``unbind()``.
    """
    runtime = make_runtime()
    runtime.register_stage(0, module)
    runtime.bind()
    runtime.zero_grad()
    with runtime.group_weights(0, 0, prepare=runtime.reduction_plan(0, (0,), final=True).before_region.get(0, ())):
        closure()
    plan = runtime.reduction_plan(0, (0,), final=True)
    for bucket_ids in plan.after_region.values():
        for bucket_id in bucket_ids:
            runtime.reduce(plan, bucket_id)
    runtime.wait()
    return runtime
