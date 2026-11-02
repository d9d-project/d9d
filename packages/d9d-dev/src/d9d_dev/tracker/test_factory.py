import sys
from unittest.mock import patch

import pytest
from d9d.tracker import WandbConfig, tracker_from_config


@pytest.mark.local
def test_factory_names_missing_extra():
    # A None entry in sys.modules makes the import of the module fail.
    with (
        patch.dict(sys.modules, {"d9d.tracker.provider.wandb.tracker": None}),
        pytest.raises(ImportError, match=r"Install the d9d\[tracker-wandb\] extra"),
    ):
        tracker_from_config(WandbConfig(project="proj"))
