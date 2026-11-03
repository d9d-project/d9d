from unittest.mock import Mock

import pytest
import torch
from d9d.loop.component.checkpointer import StateCheckpointer
from d9d.loop.config import CheckpointingConfig


@pytest.mark.local
def test_load_prepares_and_restores_state_outside_inference_mode(tmp_path, monkeypatch):
    checkpoint = tmp_path / "save-1"
    checkpoint.mkdir()
    state = Mock()
    phases = []

    def prepare():
        phases.append("prepare")
        assert not torch.is_inference_mode_enabled()

    def load(*, state_dict, checkpoint_id):
        phases.append("load")
        assert not torch.is_inference_mode_enabled()
        assert state_dict == {"state": state}
        assert checkpoint_id == checkpoint

    state.prepare_load.side_effect = prepare
    monkeypatch.setattr("d9d.loop.component.checkpointer.dcp.load", load)
    monkeypatch.setattr("d9d.loop.component.checkpointer.torch.cuda.empty_cache", lambda: None)
    checkpointer = StateCheckpointer(
        dist_context=Mock(),
        schedule=Mock(),
        config=CheckpointingConfig(save_dir=tmp_path, period_steps=1),
        gc=Mock(),
        run_name=None,
    )

    with torch.inference_mode():
        checkpointer.load_last_checkpoint(state)
        assert torch.is_inference_mode_enabled()

    assert phases == ["prepare", "load"]
