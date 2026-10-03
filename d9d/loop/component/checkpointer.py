import re
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.profiler import record_function

from d9d.core.dist_context import DistributedContext
from d9d.loop.config import CheckpointingConfig

from .garbage_collector import ManualGarbageCollector
from .job_schedule import JobSchedule

_SAVE_RE = re.compile(r"^save-(\d+)$")


def _save_iter_predicate(x: Path) -> int:
    match = _SAVE_RE.fullmatch(x.stem)
    if match is None:
        raise ValueError(f"Malformed checkpoint name ({x.stem}). Expected save-<step>.")
    return int(match.group(1))


class StateCheckpointer:
    """Manages the lifecycle of distributed training checkpoints.

    It saves and loads the job state with PyTorch Distributed Checkpoint (DCP). It names checkpoints
    by step, keeps only the latest ones and synchronizes the ranks around each save and load.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        schedule: JobSchedule,
        config: CheckpointingConfig,
        gc: ManualGarbageCollector,
        run_name: str | None,
    ):
        """Constructs the ``StateCheckpointer`` object.

        Args:
            dist_context: The distributed context.
            schedule: The job schedule that tracks the current step.
            config: The checkpointing configuration.
            gc: The garbage collector used to free memory around checkpoint I/O.
            run_name: The run name to append to the save directory. ``None`` uses the save directory as is.
        """
        self._dist_context = dist_context
        self._schedule = schedule
        self._gc = gc

        if run_name:
            self._save_dir = config.save_dir / run_name
        else:
            self._save_dir = config.save_dir

        self._config = config

    def _free_memory(self):
        self._gc.collect_forced()
        torch.cuda.empty_cache()

    def _get_sorted_checkpoint_dirs(self) -> list[Path]:
        if not self._save_dir:
            return []

        if not self._save_dir.is_dir():
            return []

        checkpoint_dirs = [x for x in self._save_dir.iterdir() if x.is_dir() and _SAVE_RE.fullmatch(x.stem)]
        checkpoint_dirs = sorted(checkpoint_dirs, key=_save_iter_predicate)
        return checkpoint_dirs

    def _next_checkpoint_id(self) -> Path:
        next_name = f"save-{self._schedule.current_step}"
        return self._save_dir / next_name

    def _purge_old_checkpoints(self):
        if not self._dist_context.is_main_process:
            return
        if not self._config.num_to_keep:
            return

        to_delete = self._get_sorted_checkpoint_dirs()[: -self._config.num_to_keep]

        for delete_dir in to_delete:
            self._dist_context.logger.info(f"Purging checkpoint {delete_dir}")
            shutil.rmtree(delete_dir)

    def _checkpoint(self, state: Stateful):
        with record_function("Checkpoint Save"):
            next_checkpoint_id = self._next_checkpoint_id()

            self._dist_context.logger.info("Freeing up memory before checkpointing")
            self._free_memory()
            self._dist_context.logger.info("Waiting for world before saving checkpoint")
            self._dist_context.wait_world()
            self._dist_context.logger.info(f"Saving checkpoint {next_checkpoint_id}")

            save_from = {"state": state}
            dcp.save(state_dict=save_from, checkpoint_id=next_checkpoint_id)

            self._purge_old_checkpoints()
            self._free_memory()

            self._dist_context.logger.info("Waiting for world after saving checkpoint")
            self._dist_context.wait_world()
            self._dist_context.logger.info("Checkpoint successfully saved across the world")

    def checkpoint_if_needed(self, state: Stateful):
        """Saves a checkpoint if the current step matches the configured period or is the last step.

        Args:
            state: The ``Stateful`` object to save.
        """
        if self._schedule.should_do_action(
            self._config.period_steps, enable_on_last_step_if_periodic=True, is_post_step_action=True
        ):
            self._checkpoint(state)

    def _last_checkpoint_id(self) -> Path | None:
        checkpoints = self._get_sorted_checkpoint_dirs()
        if len(checkpoints) == 0:
            return None
        return checkpoints[-1]

    def _load(self, state: Stateful):
        last_checkpoint = self._last_checkpoint_id()

        if last_checkpoint is None:
            self._dist_context.logger.info("Starting job from scratch")
            return

        self._dist_context.logger.info("Waiting for world before loading checkpoint")
        self._dist_context.wait_world()
        self._dist_context.logger.info(f"Loading checkpoint {last_checkpoint}")

        load_into = {"state": state}
        dcp.load(state_dict=load_into, checkpoint_id=last_checkpoint)
        self._free_memory()

        self._dist_context.logger.info("Waiting for world after loading checkpoint")
        self._dist_context.wait_world()
        self._dist_context.logger.info("Checkpoint successfully loaded across the world")

    def load_last_checkpoint(self, state: Stateful):
        """Loads the latest checkpoint in the save directory.

        If no checkpoint is found, the state stays unchanged and the job starts from scratch.

        Args:
            state: The ``Stateful`` object to load the checkpoint into.
        """
        self._load(state)
