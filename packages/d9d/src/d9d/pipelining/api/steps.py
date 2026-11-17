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
    """The event that a resumable forward yields to its executor at a boundary.

    Attributes:
        module: The module whose parameter group the next region uses, or ``None`` if the boundary only finishes a
            region.
        state: The live state at the boundary. It holds every new differentiable value that later regions need.
            Tensors exposed at earlier boundaries keep their producers and can stay outside the state.
        prefetch: The module whose parameter group the region after the next one uses, or ``None`` to leave the
            choice to the executor.
    """

    module: nn.Module | None
    state: TState
    prefetch: nn.Module | None = None

    def __await__(self) -> Generator[WindowBoundary[TState], TState, TState]:
        """Suspends the coroutine until the executor resumes this boundary.

        Returns:
            The state that the executor supplies on resume.

        Yields:
            This boundary, to the executor.
        """
        return (yield self)


class WindowSteps:
    """The boundaries of one call of a resumable forward.

    The executor creates an instance for each call, also when several microbatches run in turn. User compute
    contexts, such as ``no_grad()`` and autocast, must not span a boundary. The executor enters the compute contexts
    around each resume.
    """

    def __init__(self, *, eager: bool = False):
        """Constructs the ``WindowSteps`` object.

        Args:
            eager: Whether nested resumable modules run through their ``forward()``, so that their module hooks run.
        """
        self._active = False
        self._eager = eager

    async def enter(self, module: nn.Module, state: TState, *, prefetch: nn.Module | None = None) -> TState:
        """Finishes the open region, if any, and requests the parameter group of ``module``.

        Args:
            module: The module whose parameter group covers every parameter that the code reads before the next
                boundary.
            state: Every live differentiable value, including the values that later regions only pass through.
            prefetch: The module whose parameter group the region after this one uses, or ``None`` to leave the
                choice to the executor. All microbatches and ranks must pass the same value.

        Returns:
            The state that the executor supplies. Use it in place of ``state`` from here on.
        """
        state = await WindowBoundary(module, state, prefetch)
        self._active = True
        return state

    async def finish(self, state: TState) -> TState:
        """Finishes the open region without entering another one.

        Args:
            state: The live state that the region produced and later regions need.

        Returns:
            The state that the executor supplies. Use it in place of ``state`` from here on.

        Raises:
            ValueError: If no region is open.
        """
        if not self._active:
            raise ValueError("No window region is open. Call enter() before finish().")
        state = await WindowBoundary(None, state)
        self._active = False
        return state

    async def call(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """Runs a module between boundaries and returns its output.

        An ordinary module forms one region with its parameter group. A resumable module runs its own
        ``forward_steps()`` with these boundaries, so each of its regions requests its own group. With ``eager=True``,
        a resumable module runs through its ``forward()`` instead, so its module hooks run. Differentiable operations
        outside module calls need a region opened with ``enter()``.

        Args:
            module: The module to run.
            *args: The positional arguments of the module.
            **kwargs: The keyword arguments of the module.

        Returns:
            The output of the module, as the executor supplies it at the finishing boundary.

        Raises:
            ValueError: If a region opened with ``enter()`` is still open. Call ``finish()`` with its live state
                first.
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
        """Runs the forward pass and suspends at the boundaries of ``steps``.

        Args:
            steps: The boundaries of this call, created by the executor.
            inputs: The input of the model or pipeline stage.
            shared: The shared input of the microbatch.

        Returns:
            The output of the model or stage. When the outermost coroutine returns, the executor finishes any open
                region. A return from a nested helper coroutine finishes no region.
        """
        ...


def run_steps(
    module: ModuleSupportsWindows[TInput_contra, TShared_contra, TOutput_co],
    inputs: TInput_contra,
    shared: TShared_contra,
) -> TOutput_co:
    """Runs a resumable forward eagerly, with the ordinary autograd graph and without parameter management.

    Each boundary resumes at once with its own state. The function closes the coroutine on exit, also on failure.

    Args:
        module: The module with a resumable forward.
        inputs: The input of the model or stage.
        shared: The shared input of the microbatch.

    Returns:
        The output of the model or stage.

    Raises:
        TypeError: If the coroutine yields an object other than a ``WindowBoundary``.
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
