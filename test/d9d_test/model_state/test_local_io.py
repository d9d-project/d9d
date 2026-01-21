import json

import pytest
import torch
from d9d.model_state.io import (
    load_model_state,
    read_model_state,
    save_model_state,
    write_model_state_local,
)
from d9d.model_state.io.dto import MODEL_STATE_INDEX_FILE_NAME
from d9d.model_state.mapper import ModelStateMapper, StateGroup
from d9d.model_state.mapper.adapters import identity_mapper_from_module
from d9d.model_state.mapper.compose import ModelStateMapperSequential
from d9d.model_state.mapper.leaf import ModelStateMapperRename, ModelStateMapperStackTensors
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


class SplitTensorMapper(ModelStateMapper):
    def __init__(self, fused_name, part_a, part_b, split_dim=0):
        self.inputs = frozenset([fused_name])
        self.outputs = frozenset([part_a, part_b])
        self.fused = fused_name
        self.parts = [part_a, part_b]
        self.split_dim = split_dim

    def state_dependency_groups(self) -> frozenset[StateGroup]:
        return frozenset([StateGroup(inputs=self.inputs, outputs=self.outputs)])

    def apply(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tensor = group[self.fused]
        return {self.parts[0]: tensor[0], self.parts[1]: tensor[1]}


@pytest.mark.local
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_save_load_roundtrip_simple(tmp_path, device):
    model = SimpleModel().to(device)
    # Modify weights to be recognizable
    with torch.no_grad():
        model.fc.weight.fill_(1.5)
        model.fc.bias.fill_(0.5)

    save_dir = tmp_path / "ckpt"
    mapper = identity_mapper_from_module(model)

    save_model_state(
        dest_dir=save_dir,
        mapper=mapper,
        model=model,
        show_progress=False
    )

    # Check meta format (HF compat)
    index_content = json.loads((save_dir / MODEL_STATE_INDEX_FILE_NAME).read_text(encoding="utf-8"))
    assert index_content == {
        "metadata": {"total_size": 40},
        "weight_map": {
            "fc.weight": "model-00001-of-00001.safetensors",
            "fc.bias": "model-00001-of-00001.safetensors"
        }
    }

    # Load back
    new_model = SimpleModel().to(device)
    load_model_state(
        src_dir=save_dir,
        mapper=mapper,
        device=device,
        model=new_model,
        show_progress=False
    )

    assert torch.equal(new_model.fc.weight, model.fc.weight)
    assert torch.equal(new_model.fc.bias, model.fc.bias)


@pytest.mark.local
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_model_partial_save_load_with_renaming(tmp_path, device):
    model = SimpleModel().to(device)
    save_dir = tmp_path / "rename_ckpt"

    # We only save weight, ignore bias for this test to show partial saving works
    save_mapper = ModelStateMapperRename("fc.weight", "layer.w")
    load_mapper = ModelStateMapperRename("layer.w", "fc.weight")

    save_model_state(save_dir, save_mapper, model, show_progress=False)

    # Check meta format (HF compat)
    index_content = json.loads((save_dir / MODEL_STATE_INDEX_FILE_NAME).read_text(encoding="utf-8"))
    assert index_content == {
        "metadata": {"total_size": 32},
        "weight_map": {
            "layer.w": "model-00001-of-00001.safetensors"
        }
    }

    new_model = SimpleModel().to(device)

    # Read back
    load_model_state(
        src_dir=save_dir,
        mapper=load_mapper,
        device=device,
        model=new_model,
        show_progress=False
    )
    assert torch.equal(new_model.fc.weight, model.fc.weight)
    assert not torch.equal(new_model.fc.bias, model.fc.bias)


@pytest.mark.local
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_save_load_complex_io_transformation(tmp_path, device):
    # 1. Create source Data
    t_a = torch.ones(4, device=device)
    t_b = torch.zeros(4, device=device)
    generator = [("a", t_a), ("b", t_b)]

    # 2. Define Mapper: (a, b) -> stack -> ab_stacked
    mapper_save = ModelStateMapperStackTensors(["a", "b"], "ab_stacked", stack_dim=0)

    dest = tmp_path / "stacked"
    write_model_state_local(dest, mapper_save, generator, show_progress=False)

    # 3. Load back: Input(ab_stacked) -> Split -> Output(a, b)
    mapper_load = SplitTensorMapper("ab_stacked", "a_rec", "b_rec", split_dim=0)

    stream = read_model_state(dest, mapper_load, device=device, show_progress=False)
    res = dict(stream)

    assert res.keys() == {"a_rec", "b_rec"}
    assert torch.equal(res["a_rec"], t_a)
    assert torch.equal(res["b_rec"], t_b)


@pytest.mark.local
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sharding_enforcement(tmp_path, device):
    t_heavy = torch.zeros(1024 * 1024, dtype=torch.uint8, device=device)  # 1MB
    gen = [("t1", t_heavy), ("t2", t_heavy)]

    dest = tmp_path / "sharded"
    mapper = ModelStateMapperSequential([
        ModelStateMapperRename("t1", "t1"),
        ModelStateMapperRename("t2", "t2")
    ])  # Identity basically, ensuring groups are processed

    write_model_state_local(dest, mapper, gen, shard_size_gb=0.0015, show_progress=False)

    # Check meta format (HF compat)
    index_content = json.loads((dest / MODEL_STATE_INDEX_FILE_NAME).read_text(encoding="utf-8"))
    assert index_content == {
        "metadata": {"total_size": 2097152},
        "weight_map": {
            "t1": "model-00001-of-00002.safetensors",
            "t2": "model-00002-of-00002.safetensors"
        }
    }

    files = {x.name for x in dest.iterdir()}
    expect_files = {
        MODEL_STATE_INDEX_FILE_NAME,
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors"
    }
    assert files == expect_files
