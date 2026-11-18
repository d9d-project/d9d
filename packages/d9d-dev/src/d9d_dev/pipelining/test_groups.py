import pytest
from d9d.pipelining.api import mark_parameter_group, marked_parameter_groups
from torch import nn


class _Plain(nn.Module):
    def __init__(self):
        super().__init__()
        self.loose = nn.Linear(2, 2)
        self.block = mark_parameter_group(nn.Sequential(nn.Linear(2, 2), mark_parameter_group(nn.Linear(2, 2))))


class _Resumable(_Plain):
    async def forward_steps(self, steps, inputs):
        return await steps.call(self.block, self.loose(inputs))


@pytest.mark.local
def test_marked_parameter_groups():
    model = _Resumable()
    assert marked_parameter_groups(model) == [model.block, model.block[1]]
    assert marked_parameter_groups(nn.Linear(2, 2)) == []
    # Only a resumable forward can request the groups, so marks in another stage are a mistake.
    with pytest.raises(ValueError, match=r"stage \(_Plain\) marks parameter groups but has no forward_steps\(\)"):
        marked_parameter_groups(_Plain())
