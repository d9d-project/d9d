import json

import pytest
import torch
from d9d.core.dist_context import FLAT_DOMAIN, REGULAR_DOMAIN, DeviceMeshParameters
from d9d.model_state.io import (
    read_model_state,
    save_model_state_pipeline_parallel,
    write_model_state_distributed,
)
from d9d.model_state.io.dto import MODEL_STATE_INDEX_FILE_NAME
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import (
    ModelStateMapperIdentity,
    ModelStateMapperRename,
)
from torch import nn


@pytest.mark.distributed
def test_distributed_write_sharding(dist_ctx_factory, shared_tmp_dir):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    rank = dist_ctx.mesh_for(FLAT_DOMAIN).get_local_rank()
    group = dist_ctx.mesh_for(FLAT_DOMAIN).get_group()

    my_tensor = torch.scalar_tensor(rank, dtype=torch.float32)
    name = f"tensor_{rank}"

    # Mapper just passes it through
    mapper = ModelStateMapperIdentity(name)

    gen = [(name, my_tensor)]
    dest = shared_tmp_dir / "dist_save"

    write_model_state_distributed(
        dest_dir=dest,
        mapper=mapper,
        state_generator=gen,
        process_group=group,
        show_progress=False
    )

    dist_ctx.wait_world()

    if dist_ctx.is_main_process:
        meta = json.loads((dest / MODEL_STATE_INDEX_FILE_NAME).read_text(encoding="utf-8"))
        assert meta == {
            "metadata": {"total_size": 32},
            "weight_map": {
                "tensor_0": "model-00001-of-00008.safetensors",
                "tensor_1": "model-00002-of-00008.safetensors",
                "tensor_2": "model-00003-of-00008.safetensors",
                "tensor_3": "model-00004-of-00008.safetensors",
                "tensor_4": "model-00005-of-00008.safetensors",
                "tensor_5": "model-00006-of-00008.safetensors",
                "tensor_6": "model-00007-of-00008.safetensors",
                "tensor_7": "model-00008-of-00008.safetensors"
            }
        }

        loaded_state = read_model_state(
            dest,
            mapper=ModelStateMapperParallel([ModelStateMapperIdentity(f"tensor_{i}") for i in range(8)]),
            device="cpu",
            show_progress=False
        )
        assert dict(loaded_state) == {
            f"tensor_{i}": torch.scalar_tensor(i, dtype=torch.float32)
            for i in range(8)
        }


@pytest.mark.distributed
def test_pipeline_parallel_save(dist_ctx_factory, shared_tmp_dir):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(
        pipeline_parallel=4,
        expert_parallel=2,
        data_parallel_replicate=2
    ))
    mesh = dist_ctx.mesh_for(REGULAR_DOMAIN)
    pp_mesh = mesh["pp"]
    global_rank = pp_mesh.get_rank()

    model = nn.Linear(1, 1)  # simple placeholder
    with torch.no_grad():
        model.weight.fill_(global_rank)

    mapper = ModelStateMapperRename("weight", f"layer_{global_rank}.w")

    dest = shared_tmp_dir / "pp_save"

    save_model_state_pipeline_parallel(
        dest_dir=dest,
        mapper=mapper,
        device_mesh=mesh,
        pipeline_dim_name="pp",
        models=[model],
        show_progress=False
    )

    dist_ctx.wait_world()

    if dist_ctx.is_main_process:
        meta = json.loads((dest / MODEL_STATE_INDEX_FILE_NAME).read_text(encoding="utf-8"))
        assert meta == {
            "metadata": {"total_size": 16},
            "weight_map": {
                "layer_0.w": "model-00001-of-00004.safetensors",
                "layer_2.w": "model-00002-of-00004.safetensors",
                "layer_4.w": "model-00003-of-00004.safetensors",
                "layer_6.w": "model-00004-of-00004.safetensors"
            }
        }
        loaded_state = read_model_state(
            dest,
            mapper=ModelStateMapperParallel([ModelStateMapperIdentity(f"layer_{i}.w") for i in range(0, 8, 2)]),
            device="cpu",
            show_progress=False
        )
        assert dict(loaded_state) == {
            f"layer_{i}.w": torch.tensor([[i]], dtype=torch.float32)
            for i in range(0, 8, 2)
        }
