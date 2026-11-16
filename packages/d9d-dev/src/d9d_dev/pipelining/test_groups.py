import pytest
from d9d.pipelining.api import mark_parameter_group
from d9d.pipelining.api.groups import group_modules, parameter_group_roots
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
@pytest.mark.parametrize("resumable", [False, True])
def test_parameter_groups_partition_the_stage(resumable):
    model = _Resumable() if resumable else _Plain()

    roots = parameter_group_roots(model)

    # Marks take effect only in a resumable stage.
    assert roots == ([model, model.block, model.block[1]] if resumable else [model])
    groups = [
        [id(p) for module in group_modules(root, set(roots)) for p in module.parameters(recurse=False)]
        for root in roots
    ]
    every_parameter = [p for group in groups for p in group]
    assert sorted(every_parameter) == sorted(id(p) for p in model.parameters())
