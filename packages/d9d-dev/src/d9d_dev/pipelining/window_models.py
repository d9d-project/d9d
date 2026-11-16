from d9d.pipelining.api import mark_window, run_steps
from torch import nn


class _Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, inputs, *, skip=None):
        if skip is not None:
            inputs = inputs + skip
        return self.linear(inputs).tanh()


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = mark_window(_Block(3))
        self.last = mark_window(_Block(3))
        self.last.linear.weight = self.first.linear.weight

    def forward(self, inputs, shared):
        return run_steps(self, inputs, shared)

    def reference(self, inputs):
        skip = self.first(inputs)
        hidden = self.first(skip)
        return self.last(hidden, skip=skip)


class _Calls(_MLP):
    async def forward_steps(self, steps, inputs, shared):
        skip = await steps.call(self.first, inputs)
        hidden = await steps.call(self.first, skip)
        return await steps.call(self.last, hidden, skip=skip)


class _Manual(_MLP):
    async def forward_steps(self, steps, inputs, shared):
        inputs = await steps.enter(self.first, inputs)
        hidden = self.first(inputs)
        hidden, skip = await steps.enter(self.first, (hidden, hidden))
        hidden = self.first(hidden)
        hidden, skip = await steps.enter(self.last, (hidden, skip))
        return self.last(hidden, skip=skip)


class _Revisited(_MLP):
    def reference(self, inputs):
        return self.first(self.last(self.first(inputs)))

    async def forward_steps(self, steps, inputs, shared):
        hidden = await steps.call(self.first, inputs)
        hidden = await steps.call(self.last, hidden)
        return await steps.call(self.first, hidden)


class _LocalSkip(_MLP):
    """Reads the output of an earlier region and the stage input from local variables, not from the boundary."""

    def reference(self, inputs):
        skip = self.first(inputs)
        return self.last(self.first(skip), skip=skip + inputs)

    async def forward_steps(self, steps, inputs, shared):
        skip = await steps.call(self.first, inputs)
        hidden = await steps.call(self.first, skip)
        hidden = await steps.enter(self.last, hidden)
        return self.last(hidden, skip=skip + inputs)
