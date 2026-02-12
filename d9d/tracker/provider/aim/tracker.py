from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Self, TypedDict, cast

import torch
from aim import Distribution, Run

from d9d.tracker import BaseTracker, BaseTrackerRun, RunConfig

from .config import AimConfig


class AimState(TypedDict):
    """
    State dictionary format for persisting Aim tracker state.
    """

    restart_hash: str | None


class AimRun(BaseTrackerRun):
    """
    Active run implementation for Aim.

    Wraps the underlying `aim.Run` object to adhere to the d9d BaseTrackerRun interface.
    """

    def __init__(self, run: Run):
        self._run = run
        self._step = 0
        self._context: dict[str, str] = {}

    def set_step(self, step: int):
        self._step = step

    def set_context(self, context: dict[str, str]):
        self._context = context

    def scalar(self, name: str, value: float, context: dict[str, str] | None = None):
        if context is None:
            track_context = self._context
        else:
            track_context = {**self._context, **context}

        self._run.track(name=name, value=value, context=track_context, step=self._step)

    def bins(self, name: str, values: torch.Tensor, context: dict[str, str] | None = None):
        if context is None:
            track_context = self._context
        else:
            track_context = {**self._context, **context}

        self._run.track(
            name=name,
            value=Distribution(hist=values.numpy(), bin_range=(0, values.shape[0])),
            context=track_context,
            step=self._step,
        )


class AimTracker(BaseTracker[AimConfig]):
    """
    Aim-based tracker implementation.

    Caches the run hash to allow experiment resumption from checkpoints.
    """

    def __init__(self, config: AimConfig):
        self._config = config

        self._restart_hash: str | None = None
        self._run: Run | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = cast(AimState, state_dict)
        self._restart_hash = state["restart_hash"]

    def state_dict(self) -> dict[str, Any]:
        return {"restart_hash": self._restart_hash}

    @contextmanager
    def open(self, properties: RunConfig) -> Generator[BaseTrackerRun, None, None]:
        run = Run(
            run_hash=self._restart_hash,
            repo=self._config.repo,
            log_system_params=self._config.log_system_params,
            capture_terminal_logs=self._config.capture_terminal_logs,
            system_tracking_interval=self._config.system_tracking_interval,
        )
        run.name = properties.name
        run.description = properties.description
        run["hparams"] = properties.hparams

        self._restart_hash = run.hash
        self._run = run

        yield AimRun(run)

        self._run.close()
        self._run = None

    @classmethod
    def from_config(cls, config: AimConfig) -> Self:
        return cls(config)
