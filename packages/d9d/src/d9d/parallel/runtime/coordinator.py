"""Stage registration, binding and the gradients of a step."""

import dataclasses
from collections.abc import Collection, Iterator, Sequence
from types import TracebackType
from typing import Self

import torch
from torch import nn

from .bucket import AnyBucket, BucketGradients, GatheredBucket, PendingGradientBuffers, ReplicatedBucket
from .parameter import install_tensor_hook_bridge, make_local_handle
from .plan import StagePlan, plan_stages
from .stage import StageParameters, StageState, StepState


@dataclasses.dataclass(slots=True)
class _BoundState:
    """The resources that exist between ``bind()`` and ``unbind()``.

    Attributes:
        gather_stream: The stream that gathers weights.
        reduce_stream: The stream that reduces gradients.
        step: The state of the current step.
        pending: The gradient buffers that the reduce stream may still read.
        stages: The parameter state of each stage, by stage index.
    """

    gather_stream: torch.cuda.Stream
    reduce_stream: torch.cuda.Stream
    step: StepState
    pending: PendingGradientBuffers
    stages: dict[int, StageState] = dataclasses.field(default_factory=dict)

    def buckets(self) -> Iterator[AnyBucket]:
        """Yields the buckets of all stages."""
        for state in self.stages.values():
            yield from state.buckets


class ParameterRuntime:
    """The data parallelism that the pipeline program drives.

    Modules keep sharded parameters. Around each region of a stage, the runtime puts the unsharded parameters of its
    group into the modules. After the backward pass of a group, it reduces the gradients that are complete.

    The lifecycle is ``register_stage()`` for every stage, then ``bind()``, the steps, and ``unbind()``. A bound
    runtime can be unbound and bound again, e.g. around an offload. A step starts with ``zero_grad()`` and ends with
    ``wait()``. As a context manager, the runtime calls ``unbind()`` on exit.
    """

    def __init__(self, gather_dims: Collection[str], bucket_size_mib: int, training: bool = True):
        """Constructs the ``ParameterRuntime`` object.

        Args:
            gather_dims: The mesh dimensions along which data parallelism shards parameters. The runtime gathers
                parameters along them, and leaves every other ``Shard`` dimension to the layer.
            bucket_size_mib: The size limit of buckets without a gather, in MiB.
            training: Whether the job computes gradients.

        Raises:
            ValueError: If the bucket size is not positive.
        """
        if bucket_size_mib <= 0:
            raise ValueError(f"The bucket size bucket_size_mib ({bucket_size_mib}) must be positive.")
        self._gather_dims = frozenset(gather_dims)
        self._bucket_size_mib = bucket_size_mib
        self._training = training
        self._groups: dict[int, list[nn.Module]] = {}
        self._bound: _BoundState | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ):
        self.unbind()

    def register_stage(self, stage_index: int, module: nn.Module, groups: Sequence[nn.Module] = ()):
        """Registers a pipeline stage.

        Args:
            stage_index: The index of the stage.
            module: The stage root. Its group holds the parameters outside ``groups``.
            groups: The roots of the other parameter groups of the stage, e.g. ``marked_parameter_groups(module)``.
                Without them, the stage runs as one group.

        Raises:
            ValueError: If the runtime is bound, if the stage is already registered, or if a group root is repeated, is
                the stage root or lies outside it.
        """
        if self._bound is not None:
            raise ValueError(f"The runtime is bound, so it cannot register stage ({stage_index}). Call unbind() first.")
        if stage_index in self._groups:
            raise ValueError(f"The stage ({stage_index}) is already registered. Register each stage once.")
        submodules = set(module.modules()) - {module}
        roots: set[nn.Module] = set()
        for group in groups:
            if group in roots:
                raise ValueError(
                    f"The group root ({type(group).__name__}) of stage ({stage_index}) is repeated. Pass each root "
                    "once, e.g. marked_parameter_groups(module)."
                )
            if group not in submodules:
                raise ValueError(
                    f"The group root ({type(group).__name__}) of stage ({stage_index}) is not a submodule of the "
                    "stage. Pass the roots inside the stage root, e.g. marked_parameter_groups(module)."
                )
            roots.add(group)
        self._groups[stage_index] = [module, *groups]

    def _require_bound(self) -> _BoundState:
        if self._bound is None:
            raise ValueError("The runtime is not bound. Call bind() first.")
        return self._bound

    def bind(self):
        """Builds the unsharded parameters and the gradient buffers.

        Run the steps on the stream that runs ``bind()``.

        Raises:
            ValueError: If the runtime is already bound, if a parameter is shared by several stages, if the placements
                of a parameter are not supported, or if a shard does not match ``torch.chunk``.
        """
        if self._bound is not None:
            raise ValueError("The runtime is already bound. Call unbind() first.")

        plans = plan_stages(self._groups, self._training, self._bucket_size_mib, self._gather_dims)
        bound = _BoundState(
            gather_stream=torch.cuda.Stream(),
            reduce_stream=torch.cuda.Stream(),
            step=StepState(),
            pending=PendingGradientBuffers(),
        )
        self._bound = bound
        try:
            for stage_index, plan in plans.items():
                self._bind_stage(bound, stage_index, plan)
        except Exception:
            self.unbind()
            raise

    @staticmethod
    def _bind_stage(bound: _BoundState, stage_index: int, plan: StagePlan):
        # The stage joins the bound state before its buckets exist, so teardown releases them if a later one fails.
        state = StageState(
            stage_index=stage_index,
            group_indices={group: index for index, group in enumerate(plan.groups)},
            handles=[],
            handles_by_group=[],
            buckets=[],
            gathered_by_group=[[] for _ in plan.parameters_by_group],
            step=bound.step,
        )
        bound.stages[stage_index] = state

        # A parameter without a gather views its local data. The unsharded parameters of a gathered bucket view its
        # weight buffer, so the bucket creates them.
        gathered = {x for bucket in plan.buckets if bucket.gather is not None for x in bucket.parameters}
        handles = {x: make_local_handle(x) for x in plan.parameters if x not in gathered}

        # The buckets keep the order of the plan, which is the reduction order.
        for bucket_plan in plan.buckets:
            match bucket_plan.gather:
                case None:
                    bucket: AnyBucket = ReplicatedBucket(
                        bucket_plan, [handles[x] for x in bucket_plan.parameters], bound.reduce_stream
                    )
                case gather:
                    bucket = GatheredBucket(
                        bucket_plan, gather, bound.gather_stream, bound.reduce_stream, bound.pending
                    )
                    handles.update((handle.plan, handle) for handle in bucket.handles)
                    for group in bucket_plan.groups:
                        state.gathered_by_group[group].append(bucket)
            state.buckets.append(bucket)

        state.handles = list(handles.values())
        state.handles_by_group = [[handles[x] for x in group] for group in plan.parameters_by_group]
        for handle in state.handles:
            install_tensor_hook_bridge(handle)

    def unbind(self):
        """Releases the unsharded parameters, the gradient buffers and the hooks.

        It sets the ``.grad`` of the sharded parameters to ``None``. It does nothing on an unbound runtime.
        """
        bound = self._bound
        if bound is None:
            return
        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        torch.cuda.current_stream().wait_stream(bound.gather_stream)
        bound.pending.free_all()
        for bucket in bound.buckets():
            bucket.unbind()
        for state in bound.stages.values():
            for handle in state.handles:
                if handle.hook is not None:
                    handle.hook.remove()
                # A graph that the program still holds, e.g. through a logged loss, keeps the unsharded parameter
                # alive. So it must keep neither the gradient buffer nor the local data, which an offload replaces.
                handle.unsharded.grad = None
                handle.unsharded.data = handle.unsharded.data.new_empty(0)
            state.unbind()
        bound.step.reduced.clear()
        self._bound = None

    def stage(self, stage_index: int) -> StageParameters:
        """Returns the parameter groups of a bound stage.

        Args:
            stage_index: The index of the stage.

        Returns:
            The parameter groups of the stage.

        Raises:
            ValueError: If the runtime is not bound, or if the stage is not registered.
        """
        stages = self._require_bound().stages
        if stage_index not in stages:
            raise ValueError(f"The stage ({stage_index}) is not registered. Call register_stage() before bind().")
        return StageParameters(stages[stage_index])

    @staticmethod
    def _gradients(bound: _BoundState) -> list[BucketGradients]:
        return [bucket.gradients for bucket in bound.buckets() if bucket.gradients is not None]

    @staticmethod
    def _invalidate_weights(bound: _BoundState):
        for bucket in bound.buckets():
            match bucket:
                case GatheredBucket():
                    bucket.invalidate_weights()
                case ReplicatedBucket():
                    # The parameters view the local data, which holds the latest writes.
                    pass

    def wait(self):
        """Finishes the gradients of the step for the current stream.

        The sharded parameters run their post-accumulate-grad hooks. Call it once per step, after the final
        reductions and before the optimizer step.

        Raises:
            ValueError: If the runtime is not bound, if this step is already finished, if the backward passes did not
                reduce a bucket, if a gradient was replaced during the step, or if a post-accumulate-grad hook set a
                gradient to ``None``.
        """
        bound = self._require_bound()
        step = bound.step
        if step.finished:
            raise ValueError("The gradients of this step are already finished. Call zero_grad() before the next step.")

        for state in bound.stages.values():
            missing = sorted(
                {
                    group
                    for bucket in state.buckets
                    if bucket.plan.trainable and bucket not in step.reduced
                    for group in bucket.plan.groups
                }
            )
            if missing:
                raise ValueError(
                    f"The backward passes of stage ({state.stage_index}) did not reduce the parameter groups "
                    f"({missing}). Run a backward pass with final=True over every region of the stage before wait()."
                )

        gradients = self._gradients(bound)
        for item in gradients:
            item.check()

        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        bound.pending.free_all()
        # The optimizer writes the sharded weights after wait(), and a gather that no region used can still read them.
        self._invalidate_weights(bound)
        # From here on, a retry would run the hooks that already ran.
        step.finished = True
        for item in gradients:
            item.finish()

    @torch.no_grad()
    def scale_grads(self, factor: torch.Tensor):
        """Multiplies all gradients in place. Call it after ``wait()``.

        Args:
            factor: The scalar to multiply the gradients by.

        Raises:
            ValueError: If the runtime is not bound, or if the gradients of the step are not finished.
        """
        bound = self._require_bound()
        # Before wait(), a reduction on the reduce stream can still overwrite the scaled gradients.
        if not bound.step.finished:
            raise ValueError("The gradients of this step are not finished. Call wait() before scale_grads().")
        tensors = [item.tensor for item in self._gradients(bound)]
        if tensors:
            torch._foreach_mul_(tensors, factor)

    def zero_grad(self):
        """Starts a step: clears the gradients and drops the unsharded weights.

        Call it after the optimizer step and before the next forward pass. Without a bind, it does nothing.
        """
        bound = self._bound
        if bound is None:
            return
        # A failed step can leave reductions queued on the reduce stream.
        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        for bucket in bound.buckets():
            bucket.zero_grad()
        # A gather that started before the optimizer step holds the weights from before it.
        self._invalidate_weights(bound)
        bound.step.reduced.clear()
        bound.step.finished = False

    def invalidate_weights(self):
        """Drops the unsharded weights before a load of the sharded ones, including frozen and inference weights.

        The next group reads the new sharded weights. Without a bind, it does nothing: ``bind()`` reads them anew.
        """
        if self._bound is not None:
            self._invalidate_weights(self._bound)
