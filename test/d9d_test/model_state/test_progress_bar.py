from unittest.mock import patch

import pytest
import torch
from d9d.core.dist_context import REGULAR_DOMAIN, DeviceMeshParameters
from d9d.model_state.io import load_model_state, save_model_state
from d9d.model_state.io.reader import _StateLoadingFlow  # noqa: PLC2701
from d9d.model_state.io.writer import _StateWritingFlowLocal  # noqa: PLC2701
from d9d.model_state.mapper.adapters import identity_mapper_from_module
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


@pytest.mark.distributed
def test_load_model_state_tqdm_per_rank_config(dist_ctx_factory, shared_tmp_dir):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    local_rank = torch.cuda.current_device()
    device = f"cuda:{local_rank}"

    save_dir = shared_tmp_dir / "ckpt"

    if dist_ctx.is_main_process:
        model = SimpleModel().to(device)
        save_model_state(dest_dir=save_dir, mapper=identity_mapper_from_module(model), model=model, show_progress=False)

    dist_ctx.wait_world()

    pbar_state: dict = {}

    def _capture(*args, **kwargs):
        instance = _StateLoadingFlow(*args, **kwargs)
        # have to make snapshot, because after load disable=True and assert fails
        pbar_state.update(
            desc=instance._pbar.desc,
            disable=instance._pbar.disable,
            leave=instance._pbar.leave,
            total=instance._pbar.total,
        )
        return instance

    with patch("d9d.model_state.io.reader._StateLoadingFlow", side_effect=_capture):
        model = SimpleModel().to(device)
        load_model_state(
            src_dir=save_dir,
            mapper=identity_mapper_from_module(model),
            device=device,
            model=model,
            show_progress=True,
            position=local_rank,
        )

    assert pbar_state["desc"] == f"Loading Model States [{local_rank}]"
    assert pbar_state["leave"] is True
    assert pbar_state["disable"] is False
    assert pbar_state["total"] > 0


@pytest.mark.distributed
def test_save_model_state_tqdm_per_rank_config(dist_ctx_factory, shared_tmp_dir):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))
    local_rank = torch.cuda.current_device()
    device = f"cuda:{local_rank}"
    group = dist_ctx.mesh_for(REGULAR_DOMAIN)["dp_replicate"].get_group()

    pbar_state: dict = {}

    def _capture(*args, **kwargs):
        instance = _StateWritingFlowLocal(*args, **kwargs)
        pbar_state.update(
            desc=instance._pbar.desc,
            disable=instance._pbar.disable,
            leave=instance._pbar.leave,
            total=instance._pbar.total,
        )
        return instance

    from d9d.model_state.io import write_model_state_distributed
    from d9d.model_state.mapper.adapters import identity_mapper_from_module

    model = SimpleModel().to(device)
    with patch("d9d.model_state.io.writer._StateWritingFlowLocal", side_effect=_capture):
        write_model_state_distributed(
            dest_dir=shared_tmp_dir / "ckpt_save",
            mapper=identity_mapper_from_module(model),
            state_generator=model.state_dict().items(),
            process_group=group,
            show_progress=True,
            position=local_rank,
        )

    assert pbar_state["desc"] == f"Saving Model States [{local_rank}]"
    assert pbar_state["leave"] is True
    assert pbar_state["disable"] is False
    assert pbar_state["total"] > 0
