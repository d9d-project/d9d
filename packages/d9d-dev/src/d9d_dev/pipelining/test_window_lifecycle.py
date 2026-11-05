from contextlib import contextmanager, nullcontext
from weakref import ref

import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DATA_SHARD_DIMS, DenseDomainConfig, DeviceMeshParameters
from d9d.parallel.runtime import BucketId, ParameterRuntime, ReductionPlan
from d9d.parallel.runtime.plan import plan_stages
from d9d.pipelining.api import PipelineStageInfo, mark_window, run_steps
from d9d.pipelining.api.groups import group_modules, window_groups
from d9d.pipelining.runtime.offline import OfflinePipelineExecutor
from d9d.pipelining.stage import PipelineStage
from d9d.pipelining.stage.windows import WindowChain, backward_chains, forward_chains
from torch import nn
from torch.distributed.tensor import distribute_tensor
from torch.utils.checkpoint import checkpoint

from d9d_dev.nn.helper.tolerances import forward_tolerance_for
from d9d_dev.parallel.runtime_step import place_hsdp
from d9d_dev.pipelining.step_handler import FunctionStepHandler


class _RecordingRuntime:
    def __init__(self, module=None):
        self.bindings = []
        self.plans = []
        self.reductions = []
        self.modules = {} if module is None else {0: module}
        self.orders = {}

    @contextmanager
    def group_weights(self, stage_index, group, prepare=(), next_group=None):
        self.bindings.append((stage_index, group))
        yield

    def reduction_plan(self, stage_index, order, final):
        self.plans.append((stage_index, final))
        return ReductionPlan(final, {0: (BucketId(stage_index, 0),)})

    def reduce(self, plan, bucket_id):
        self.reductions.append((bucket_id, plan.final))

    def register_stage(self, stage_index, module):
        self.modules[stage_index] = module

    def group_index(self, stage_index, module):
        return window_groups(self.modules[stage_index]).index(module)

    def group_parameters(self, stage_index, group):
        if group is None:
            return []
        groups = window_groups(self.modules[stage_index])
        return list(
            {
                id(p): p
                for module in group_modules(groups[group], set(groups))
                for p in module.parameters(recurse=False)
                if p.requires_grad
            }.values()
        )

    def expected_group(self, stage_index, position):
        order = self.orders.get(stage_index, ())
        return order[position] if position < len(order) else None

    def record_order(self, stage_index, order):
        assert self.orders.setdefault(stage_index, order) == order


def _forward(chain, runtime, inputs, shared):
    chain.start_forward(inputs, shared)
    return forward_chains([chain], runtime, 0, [nullcontext()])[0]


def _backward_full(chain, runtime, loss, plan=None):
    chain.start_backward([loss], None)
    backward_chains([chain], runtime, 0, plan, [nullcontext()])


def _backward_input(chain, loss):
    chain.start_backward([loss], None)
    chain.backward_input()


class _Linear(nn.Linear):
    def forward(self, activations, constants):
        return super().forward(activations)


class _OptionalLinear(nn.Linear):
    def forward(self, activations, constants):
        return activations, super().forward(activations)


class _SelectOptional(nn.Module):
    def forward(self, activations, use_optional):
        return activations[1] if use_optional else activations[0]


class _OptionalConsumer(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = mark_window(_Linear(4, 4))
        self.optional = mark_window(_OptionalLinear(4, 4))
        self.select = mark_window(_SelectOptional())

    async def forward_steps(self, steps, inputs, shared):
        hidden = await steps.call(self.first, inputs, shared)
        hidden = await steps.call(self.optional, hidden, shared)
        return await steps.call(self.select, hidden, shared)

    def forward(self, inputs, shared):
        return run_steps(self, inputs, shared)


@pytest.mark.local
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("use_optional", [False, True])
def test_unused_groups_keep_parameter_bindings_in_order(split, use_optional):
    model = _OptionalConsumer()
    inputs = torch.randn(2, 4, requires_grad=True)
    model(inputs, use_optional).sum().backward()
    expected = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    expected_input = inputs.grad.clone()
    model.zero_grad(set_to_none=True)
    inputs.grad = None

    runtime = _RecordingRuntime(model)
    chain = WindowChain(model, 0, runtime)
    loss = _forward(chain, runtime, inputs, use_optional).sum()
    runtime.bindings.clear()
    if split:
        _backward_input(chain, loss)
        assert all(p.grad is None for p in model.parameters())
        assert runtime.bindings == [(0, index) for index in (3, 2, 1)]
        runtime.bindings.clear()
        chain.backward_weight(None)
    else:
        _backward_full(chain, runtime, loss)

    assert runtime.bindings == [(0, index) for index in (3, 2, 1)]
    torch.testing.assert_close(chain.input_grads([inputs])[0], expected_input)
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


class _Embedding(nn.Embedding):
    def forward(self, activations, constants):
        return super().forward(activations)


class _ParameterOnlyRegion(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = mark_window(_Embedding(4, 4))
        self.projection = mark_window(_Linear(4, 4))

    async def forward_steps(self, steps, inputs, shared):
        hidden = await steps.call(self.embedding, inputs, shared)
        return await steps.call(self.projection, hidden, shared)


@pytest.mark.local
def test_split_backward_defers_region_without_differentiable_inputs():
    model = _ParameterOnlyRegion()
    inputs = torch.tensor([0, 2])
    run_steps(model, inputs, None).sum().backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)

    runtime = _RecordingRuntime(model)
    chain = WindowChain(model, 0, runtime)
    _backward_input(chain, _forward(chain, runtime, inputs, None).sum())
    assert all(p.grad is None for p in model.parameters())
    chain.backward_weight(None)

    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


class _ScalarParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, inputs, shared):
        return self.weight


@pytest.mark.local
@pytest.mark.parametrize("loss_is_parameter", [False, True])
def test_offline_schedule_reduces_each_microbatch_and_marks_only_the_last_final(loss_is_parameter):
    model = _ScalarParameter() if loss_is_parameter else _OptionalConsumer()
    runtime = _RecordingRuntime()
    runtime.register_stage(0, model)
    schedule = OfflinePipelineExecutor(model, do_backward=True, runtime=runtime)

    schedule.step(
        (torch.randn(2, 4), torch.randn(2, 4)),
        (True, False),
        FunctionStepHandler(lambda output, _: output if loss_is_parameter else output.sum()),
    )

    assert runtime.plans == [(0, False), (0, True)]
    assert runtime.reductions == [(BucketId(0, 0), False), (BucketId(0, 0), True)]
    if loss_is_parameter:
        assert model.weight.requires_grad
        torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 2))


@pytest.mark.local
def test_groups_include_frozen_parameters():
    model = _OptionalConsumer()
    model.optional.requires_grad_(False)
    plan = plan_stages({0: model}, None, True, 32, frozenset())[0]
    assert {id(p.sharded) for p in plan.parameters} == {id(p) for p in model.parameters()}


@pytest.mark.local
def test_shared_input_cannot_bypass_boundaries_with_gradients():
    model = _OptionalConsumer()
    chain = WindowChain(model, 0, _RecordingRuntime(model))
    try:
        with pytest.raises(ValueError, match="outside a window region"):
            chain.start_forward(torch.randn(2, 4), torch.ones((), requires_grad=True))
    finally:
        chain.close()


@pytest.mark.distributed
@pytest.mark.parametrize(
    "mesh",
    [
        DeviceMeshParameters(data_parallel=8),
        DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)),
        DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=4)),
    ],
)
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_rank_dependent_unused_groups_reduce_like_one_model(dist_ctx_factory, mesh, split, tied):
    dist_ctx = dist_ctx_factory(mesh)
    model = _OptionalConsumer().cuda()
    reference = _OptionalConsumer().cuda()
    if tied:
        model.optional.weight = model.first.weight
        reference.optional.weight = reference.first.weight
    reference.load_state_dict(model.state_dict())
    place_hsdp(model, dist_ctx)
    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=32, memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()

    def inputs(rank, microbatch):
        return torch.full((2, 4), (rank + microbatch + 1) / 8, device="cuda", requires_grad=True)

    try:
        for microbatch in range(2):
            chain = WindowChain(model, 0, runtime)
            loss = _forward(chain, runtime, inputs(dist.get_rank(), microbatch), dist.get_rank() % 2 == 0).sum()
            plan = microbatch == 1
            if split:
                _backward_input(chain, loss)
                chain.backward_weight(plan)
            else:
                _backward_full(chain, runtime, loss, plan)
        runtime.wait()

        for rank in range(dist.get_world_size()):
            for microbatch in range(2):
                reference(inputs(rank, microbatch), rank % 2 == 0).sum().backward()
        for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
            gradient = distribute_tensor(expected.grad, parameter.device_mesh, parameter.placements).to_local()
            tolerance = forward_tolerance_for(torch.float32)
            torch.testing.assert_close(parameter.grad.to_local(), gradient, rtol=tolerance.rtol, atol=tolerance.atol)
    finally:
        runtime.unbind()


class _TrackedLinear(nn.Linear):
    def __init__(self, checkpointing, outputs):
        super().__init__(4, 4)
        self._checkpointing = checkpointing
        self._outputs = outputs

    def forward(self, activations, constants):
        if self._checkpointing:
            hidden_states = checkpoint(super().forward, activations, use_reentrant=False)
        else:
            hidden_states = super().forward(activations)
        output = torch.tanh(hidden_states)
        self._outputs.append(ref(output))
        return output


class _ThreeLayers(nn.Module):
    def __init__(self, checkpointing):
        super().__init__()
        self.outputs = []
        self.layers = nn.ModuleList(mark_window(_TrackedLinear(checkpointing, self.outputs)) for _ in range(3))

    def forward(self, inputs, shared):
        return run_steps(self, inputs, shared)

    async def forward_steps(self, steps, inputs, shared):
        for layer in self.layers:
            inputs = await steps.call(layer, inputs, shared)
        return inputs


@pytest.mark.local
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_backward_releases_the_activations_of_finished_regions(split, checkpointing):
    model = _ThreeLayers(checkpointing)
    inputs = torch.randn(2, 4, requires_grad=True)
    model(inputs, None).square().sum().backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    expected_input = inputs.grad.clone()
    model.zero_grad(set_to_none=True)
    inputs.grad = None
    model.outputs.clear()

    runtime = _RecordingRuntime()
    runtime.register_stage(0, model)
    chain = WindowChain(model, 0, runtime)
    loss = _forward(chain, runtime, inputs, None).square().sum()
    alive_in_first_layer = []

    def record(_grad):
        alive_in_first_layer.extend(reference() is not None for reference in model.outputs[1:])

    # The hook runs during the backward pass of the first layer, after the later layers finished theirs.
    model.layers[0].weight.register_hook(record)
    chain.start_backward([loss], None)
    # Let the chain own the loss graph, as in the pipeline's delayed weight phase.
    del loss
    if split:
        chain.backward_input()
        assert all(parameter.grad is None for parameter in model.parameters())
        input_grads = chain.input_grads([inputs])
        torch.testing.assert_close(input_grads[0], expected_input)
        # The chain waits for the weight phase, but the input gradient goes to the previous stage before it.
        sent = ref(input_grads[0])
        del input_grads
        assert sent() is None
        chain.backward_weight(None)
    else:
        backward_chains([chain], runtime, 0, None, [nullcontext()])
        torch.testing.assert_close(chain.input_grads([inputs])[0], expected_input)

    assert alive_in_first_layer == [False, False]
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


@pytest.mark.local
@pytest.mark.parametrize("execution", ["offline", "stage", "window"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_full_backward_releases_final_region_inputs_before_earlier_regions(execution, checkpointing):
    model = _ThreeLayers(checkpointing)
    inputs = tuple(torch.randn(2, 4) for _ in range(2 if execution == "window" else 1))
    for microbatch in inputs:
        model(microbatch, None).square().sum().backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    model.outputs.clear()

    final_inputs, alive_in_first_layer = [], []

    def remember_inputs(_module, args):
        final_inputs.append(ref(args[0]))

    def record(_grad):
        alive_in_first_layer.extend(reference() is not None for reference in final_inputs)

    model.layers[-1].register_forward_pre_hook(remember_inputs)
    model.layers[0].weight.register_hook(record)
    runtime = _RecordingRuntime()
    if execution == "offline":
        runtime.register_stage(0, model)
        schedule = OfflinePipelineExecutor(model, do_backward=True, runtime=runtime)
        schedule.step(inputs, (None,) * len(inputs), FunctionStepHandler(lambda output, _: output.square().sum()))
    else:
        stage = PipelineStage(
            info=PipelineStageInfo(num_stages=1, current_stage=0),
            module=model,
            group=None,
            stage_to_host_topology={0: 0},
            runtime=runtime,
        )
        stage.configure_buffers(has_backward=True, pipeline_inputs_per_microbatch=inputs)
        for microbatch, value in enumerate(inputs):
            stage.forward_one_chunk(microbatch, value, None)
        losses = [stage.get_pipeline_output(microbatch).square().sum() for microbatch in range(len(inputs))]
        if execution == "window":
            stage.backward_window(list(range(len(inputs))), losses, [nullcontext() for _ in inputs])
        else:
            stage.backward_one_chunk(0, loss=losses[0])

    assert alive_in_first_layer == [False] * len(inputs) ** 2
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


class _Ordered(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = mark_window(nn.Linear(3, 3, device="cuda"))
        self.second = mark_window(nn.Linear(3, 3, device="cuda"))
        self.calls = [self.first, self.second]

    async def forward_steps(self, steps, inputs, shared):
        for module in self.calls:
            inputs = await steps.call(module, inputs)
        return inputs

    def forward(self, inputs, shared):
        return run_steps(self, inputs, shared)


@pytest.mark.local
@pytest.mark.parametrize(
    ("forwards", "message"),
    [
        (lambda model: [[model.first, nn.Identity(), model.second]], "not registered"),
        (lambda model: [[model.first]], "omitted parameters"),
        (lambda model: [[model.first, model.second], [model.second, model.first]], "changed its parameter group order"),
        (lambda model: [[model.first, model.second], [model.first]], "parameter group order changed"),
    ],
    ids=["unregistered", "omitted", "reordered", "shortened"],
)
def test_window_rejects_an_invalid_group_order(forwards, message):
    model = _Ordered()
    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=32)
    runtime.register_stage(0, model)
    runtime.bind()
    *valid, invalid = forwards(model)
    try:
        for calls in valid:
            model.calls = calls
            chain = WindowChain(model, 0, runtime)
            chain.start_forward(torch.randn(2, 3, device="cuda"), None)
            forward_chains([chain], runtime, 0, [nullcontext()])
        model.calls = invalid
        chain = WindowChain(model, 0, runtime)
        with pytest.raises(ValueError, match=message):
            chain.start_forward(torch.randn(2, 3, device="cuda"), None)
            forward_chains([chain], runtime, 0, [nullcontext()])
    finally:
        runtime.unbind()
