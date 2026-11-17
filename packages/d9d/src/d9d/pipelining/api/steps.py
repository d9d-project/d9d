from __future__ import annotations

import dataclasses
from collections.abc import Generator
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, cast, runtime_checkable

from torch import nn

TState = TypeVar("TState")
TParams = ParamSpec("TParams")
TOutput = TypeVar("TOutput")
TOutput_co = TypeVar("TOutput_co", covariant=True)


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardBoundary(Generic[TState]):
    """The event that a resumable forward yields to its executor at a boundary.

    Attributes:
        module: The module whose parameter group the next region uses, or ``None`` if the boundary only finishes a
            region.
        state: The values that cross the boundary. It holds every differentiable value that the finished region
            computed and a later region reads. A value that crossed an earlier boundary can stay in a local variable.
        prefetch: The module whose parameter group the region after the next one uses, or ``None`` to leave the
            choice to the executor.
    """

    module: nn.Module | None
    state: TState
    prefetch: nn.Module | None = None

    def __await__(self) -> Generator[ForwardBoundary[TState], TState, TState]:
        """Suspends the coroutine until the executor resumes this boundary.

        Returns:
            The state that the executor supplies on resume.

        Yields:
            This boundary, to the executor.
        """
        return (yield self)


class ForwardSteps:
    """The boundaries of one call of a resumable forward.

    Do not keep it after ``forward_steps()`` returns. Grad mode, inference mode and autocast belong to the thread, so
    open and close their contexts between two boundaries.
    """

    def __init__(self, *, eager: bool = False):
        """Constructs the ``ForwardSteps`` object.

        Args:
            eager: Whether nested resumable modules run through their ``forward()``, so that their module hooks run.
        """
        self._active = False
        self._eager = eager

    async def enter(self, module: nn.Module, state: TState, *, prefetch: nn.Module | None = None) -> TState:
        """Finishes the open region, if any, and requests the parameter group of ``module``.

        Args:
            module: The stage root or a module marked with ``mark_parameter_group()``. Its parameter group covers every
                parameter that the code reads before the next boundary.
            state: Every differentiable value that the open region computed and a later region reads, also if the
                next region does not read it. A value that crossed an earlier boundary can stay in a local variable.
            prefetch: The module whose parameter group the region after this one uses, or ``None`` to leave the
                choice to the executor. All microbatches and ranks must pass the same value.

        Returns:
            The state that the executor supplies. Use it in place of ``state`` from here on.
        """
        state = await ForwardBoundary(module, state, prefetch)
        self._active = True
        return state

    async def finish(self, state: TState) -> TState:
        """Finishes the open region without entering another one.

        Args:
            state: Every differentiable value that the region computed and a later region reads.

        Returns:
            The state that the executor supplies. Use it in place of ``state`` from here on.

        Raises:
            ValueError: If no region is open.
        """
        if not self._active:
            raise ValueError("No region is open. Call enter() before finish().")
        state = await ForwardBoundary(None, state)
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
            ValueError: If a region opened with ``enter()`` is still open. Call ``finish()`` first.
        """
        if self._active:
            raise ValueError("A region is still open. Call finish() before call().")
        if not self._eager and isinstance(module, ModuleSupportsResumableForward):
            result = await module.forward_steps(self, *args, **kwargs)
            return await self.finish(result) if self._active else result
        args, kwargs = await self.enter(module, (args, kwargs))
        result = module(*args, **kwargs)
        return await self.finish(result)


@runtime_checkable
class ModuleSupportsResumableForward(Protocol[TParams, TOutput_co]):
    """Protocol for modules whose forward pass can suspend at execution boundaries."""

    async def forward_steps(self, steps: ForwardSteps, *args: TParams.args, **kwargs: TParams.kwargs) -> TOutput_co:
        """Runs the forward pass and suspends at the boundaries of ``steps``.

        When this coroutine returns, the executor finishes the open region, if any. A nested helper coroutine
        finishes no region when it returns.

        Args:
            steps: The boundaries of this call, created by the executor.
            *args: The positional arguments of the forward pass.
            **kwargs: The keyword arguments of the forward pass.

        Returns:
            The output of the forward pass.
        """
        ...


def run_steps(
    module: ModuleSupportsResumableForward[TParams, TOutput], *args: TParams.args, **kwargs: TParams.kwargs
) -> TOutput:
    """Runs a resumable forward eagerly, with the ordinary autograd graph and without parameter management.

    Each boundary resumes at once with its own state. Nested resumable modules run through their ``forward()``, so
    their module hooks run.

    Args:
        module: The module with a resumable forward.
        *args: The positional arguments of ``forward_steps()``, after ``steps``.
        **kwargs: The keyword arguments of ``forward_steps()``.

    Returns:
        The output of the forward pass.

    Raises:
        TypeError: If the coroutine yields an object other than a ``ForwardBoundary``.
    """
    coroutine = module.forward_steps(ForwardSteps(eager=True), *args, **kwargs)
    state: Any = None
    try:
        while True:
            try:
                boundary = coroutine.send(state)
            except StopIteration as completed:
                return cast(TOutput, completed.value)
            if not isinstance(boundary, ForwardBoundary):
                raise TypeError("forward_steps() must yield only ForwardBoundary events. Use ForwardSteps operations.")
            state = boundary.state
    finally:
        coroutine.close()
