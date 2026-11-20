"""Stage registration, binding and the gradients of a step."""

import dataclasses
from collections.abc import Collection, Sequence
from types import TracebackType
from typing import Self

import torch
from torch import nn

from .bucket import BucketGradients, PendingGradientBuffers
from .plan import plan_stages
from .stage import StageParameters, bind_stage


@dataclasses.dataclass(slots=True)
class _BoundState:
    """The resources that exist between ``bind()`` and ``unbind()``.

    Attributes:
        gather_stream: The stream that gathers weights.
        reduce_stream: The stream that reduces gradients.
        pending: The gradient buffers that the reduce stream may still read.
        stages: The parameters of each stage, by stage index.
        finished: Whether ``wait()`` finished the gradients of the current step.
    """

    gather_stream: torch.cuda.Stream
    reduce_stream: torch.cuda.Stream
    pending: PendingGradientBuffers
    stages: dict[int, StageParameters] = dataclasses.field(default_factory=dict)
    finished: bool = False

    def gradients(self) -> list[BucketGradients]:
        """Returns the gradients of the trainable buckets of all stages."""
        return [item for stage in self.stages.values() for item in stage.gradients]


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
            ValueError: If the runtime is already bound, if a parameter is shared by several stages, if a trainable
                parameter has no ``grad_dtype``, if the placements of a parameter are not supported, or if a shard
                does not match ``torch.chunk``.
        """
        if self._bound is not None:
            raise ValueError("The runtime is already bound. Call unbind() first.")

        plans = plan_stages(self._groups, self._training, self._bucket_size_mib, self._gather_dims)
        bound = _BoundState(
            gather_stream=torch.cuda.Stream(),
            reduce_stream=torch.cuda.Stream(),
            pending=PendingGradientBuffers(),
        )
        self._bound = bound
        try:
            for stage_index, plan in plans.items():
                bound.stages[stage_index] = bind_stage(
                    stage_index, plan, bound.gather_stream, bound.reduce_stream, bound.pending
                )
        except Exception:
            self.unbind()
            raise

    def unbind(self):
        """Releases the unsharded parameters, the buffers and the hooks.

        It sets the ``.grad`` of the sharded parameters to ``None``. The ``StageParameters`` of the stages raise
        afterwards. Without a bind, it does nothing.
        """
        bound = self._bound
        if bound is None:
            return
        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        torch.cuda.current_stream().wait_stream(bound.gather_stream)
        bound.pending.free_all()
        for stage in bound.stages.values():
            stage.close()
        self._bound = None

    def stage(self, stage_index: int) -> StageParameters:
        """Returns the parameters of a stage.

        Args:
            stage_index: The index of the stage.

        Returns:
            The parameters of the stage, valid until ``unbind()``.

        Raises:
            ValueError: If the runtime is not bound, or if the stage is not registered.
        """
        stages = self._require_bound().stages
        if stage_index not in stages:
            raise ValueError(f"The stage ({stage_index}) is not registered. Call register_stage() before bind().")
        return stages[stage_index]

    def wait(self):
        """Finishes the gradients of the step for the current stream.

        The sharded parameters run their post-accumulate-grad hooks. Call it once per step, after the final
        reductions and before the optimizer step.

        Raises:
            ValueError: If the runtime is not bound, if this step is already finished, if the backward passes did not
                reduce a group, if a gradient was replaced during the step, or if a post-accumulate-grad hook set a
                gradient to ``None``.
        """
        bound = self._require_bound()
        if bound.finished:
            raise ValueError("The gradients of this step are already finished. Call zero_grad() before the next step.")

        for stage_index, stage in bound.stages.items():
            missing = stage.unreduced_groups()
            if missing:
                raise ValueError(
                    f"The backward passes of stage ({stage_index}) did not reduce the parameter groups ({missing}). "
                    "Run a backward pass with final=True over every region of the stage before wait()."
                )

        gradients = bound.gradients()
        for item in gradients:
            item.check()

        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        bound.pending.free_all()
        # The optimizer writes the sharded weights after wait(), and a gather that no region used can still read them.
        for stage in bound.stages.values():
            stage.invalidate_weights()
        # From here on, a retry would run the hooks that already ran.
        bound.finished = True
        for item in gradients:
            item.finish()

    @torch.no_grad()
    def scale_grads(self, factor: torch.Tensor):
        """Multiplies all gradients in place.

        Call it after ``wait()``.

        Args:
            factor: The scalar to multiply the gradients by.

        Raises:
            ValueError: If the runtime is not bound, or if the gradients of the step are not finished.
        """
        bound = self._require_bound()
        # Before wait(), a reduction on the reduce stream can still overwrite the scaled gradients.
        if not bound.finished:
            raise ValueError("The gradients of this step are not finished. Call wait() before scale_grads().")
        tensors = [item.tensor for item in bound.gradients()]
        if tensors:
            torch._foreach_mul_(tensors, factor)

    def zero_grad(self):
        """Starts a step: zeroes the gradients and drops the unsharded weights.

        Call it after the optimizer step and before the next forward pass. Without a bind, it does nothing.
        """
        bound = self._bound
        if bound is None:
            return
        # A failed step can leave reductions queued on the reduce stream.
        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        for stage in bound.stages.values():
            stage.zero_grad()
            # A gather that started before the optimizer step holds the weights from before it.
            stage.invalidate_weights()
        bound.finished = False

    def invalidate_weights(self):
        """Drops the unsharded weights, so the next region gathers the sharded ones anew.

        Call it after a write to the sharded weights outside ``wait()`` and ``zero_grad()``, e.g. a checkpoint load.
        Without a bind, it does nothing.
        """
        if self._bound is not None:
            for stage in self._bound.stages.values():
                stage.invalidate_weights()
