import pytest
import torch
from d9d.loop.config import DtypeName, dtype_from_name


@pytest.mark.local
@pytest.mark.parametrize("name", list(DtypeName))
def test_dtype_from_name_covers_every_name(name):
    assert dtype_from_name(name) == getattr(torch, name.value)
