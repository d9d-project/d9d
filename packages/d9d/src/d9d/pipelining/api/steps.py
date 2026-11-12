from __future__ import annotations

import dataclasses
from collections.abc import Generator
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from torch import nn

TState = TypeVar("TState")
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TShared_contra = TypeVar("TShared_contra", contravariant=True)
TOutput_co = TypeVar("TOutput_co", covariant=True)


@dataclasses.dataclass(frozen=True, slots=True)
class WindowBoundary(Generic[TState]):
    """A suspension that finishes a region and optionally requests another module's parameters.

    Attributes:
        module: The module whose parameter group the next region uses, or ``None`` to only finish a region.
        state: The state exposed at this boundary. Include new differentiable values that later regions need.
            Previously exposed boundary tensors keep their producers and may be retained separately.
        prefetch: The group requested after this region, or ``None`` to use the learned order.
    """

    module: nn.Module | None
    state: TState
    prefetch: nn.Module | None = None

    def __await__(self) -> Generator[WindowBoundary[TState], TState, TState]:
        """Yields the boundary and returns the state supplied by the executor.

        Returns:
            The state supplied when the executor resumes this boundary.
        """
        return (yield self)


class WindowSteps:
    """Execution boundaries for one invocation of a resumable forward.

    A model receives this object from its executor. Each invocation has its own instance, including when several
    microbatches run in turn. User compute contexts must not span a boundary. The executor owns the contexts around
    resumes.
    """

    def __init__(self, *, eager: bool = False):
        """Constructs the ``WindowSteps`` object.

        Args:
            eager: Whether nested resumable modules run through their synchronous ``forward`` and module hooks.
        """
        self._active = False
        self._eager = eager

    async def enter(self, module: nn.Module, state: TState, *, prefetch: nn.Module | None = None) -> TState:
        """Finishes the previous region and requests the module's parameters before continuing.

        Args:
            module: The module whose parameter group covers every parameter used before the next boundary.
            state: All live differentiable state, including values passed through to later regions.
            prefetch: The next group, or ``None`` to use the learned order. All microbatches and ranks must agree.

        Returns:
            The state supplied by the executor. Reassign it before continuing the computation.
        """
        state = await WindowBoundary(module, state, prefetch)
        self._active = True
        return state

    async def finish(self, state: TState) -> TState:
        """Finishes the open region without entering another one.

        Args:
            state: The live state produced in this region that later regions need.

        Returns:
            The state supplied by the executor. Reassign it before continuing the computation.

        Raises:
            ValueError: If no region is open.
        """
        if not self._active:
            raise ValueError("No window region is open. Call enter() before finish().")
        state = await WindowBoundary(None, state)
        self._active = False
        return state

    async def call(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """Runs a module between entry and finish boundaries.

        Differentiable operations outside module calls require an explicit region opened with ``enter()``.
        An ordinary module forms one region of its parameter group. A resumable module runs its own
        ``forward_steps()`` with these boundaries, so its regions request their own groups. In eager mode it runs
        through its synchronous module call, preserving module hooks.

        Args:
            module: The module to call.
            *args: The positional arguments of the module call.
            **kwargs: The keyword arguments of the module call.

        Returns:
            The module output, after the executor processes the finish boundary.

        Raises:
            ValueError: If a manual region is still open. Finish it with its complete live state first.
        """
        if self._active:
            raise ValueError("A window region is still open. Call finish() with its live state before call().")
        if not self._eager and isinstance(module, ModuleSupportsWindows):
            result = await module.forward_steps(self, *args, **kwargs)
            return await self.finish(result) if self._active else result
        args, kwargs = await self.enter(module, (args, kwargs))
        result = module(*args, **kwargs)
        return await self.finish(result)


@runtime_checkable
class ModuleSupportsWindows(Protocol[TInput_contra, TShared_contra, TOutput_co]):
    """Protocol for modules whose forward pass can suspend at execution boundaries."""

    async def forward_steps(self, steps: WindowSteps, inputs: TInput_contra, shared: TShared_contra) -> TOutput_co:
        """Computes the model, yielding boundaries through ``steps``.

        Args:
            steps: The boundaries for this invocation, supplied by the executor.
            inputs: The input of this model or pipeline stage.
            shared: The shared input of the microbatch.

        Returns:
            The output of the model or stage. The executor finishes any open region when its outer coroutine returns.
            Returning from a nested helper coroutine does not finish a region.
        """
        ...


def run_steps(
    module: ModuleSupportsWindows[TInput_contra, TShared_contra, TOutput_co],
    inputs: TInput_contra,
    shared: TShared_contra,
) -> TOutput_co:
    """Runs a resumable forward eagerly, without graph cuts or parameter management.

    Each boundary resumes immediately with its original state. The coroutine is closed on exit, including on failure.

    Args:
        module: The model with a resumable forward.
        inputs: The input of the model or stage.
        shared: The shared input of the microbatch.

    Returns:
        The output of the model or stage, with the ordinary autograd graph intact.

    Raises:
        TypeError: If the coroutine yields anything other than a ``WindowBoundary``.
    """
    coroutine = module.forward_steps(WindowSteps(eager=True), inputs, shared)
    state: Any = None
    try:
        while True:
            try:
                boundary = coroutine.send(state)
            except StopIteration as completed:
                return cast(TOutput_co, completed.value)
            if not isinstance(boundary, WindowBoundary):
                raise TypeError("forward_steps() must yield only WindowBoundary events. Use WindowSteps operations.")
            state = boundary.state
    finally:
        coroutine.close()
