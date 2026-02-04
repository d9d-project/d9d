import re
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful

from d9d.core.dist_context import DistributedContext
from d9d.loop.config import CheckpointingConfig

from .garbage_collector import ManualGarbageCollector
from .stepper import Stepper

# TODO feat(max): async checkpointing may break everything up, but I guess we still have to support it

_SAVE_RE = re.compile(r"^save-(\d+)$")


def _save_iter_predicate(x: Path) -> int:
    match = _SAVE_RE.fullmatch(x.stem)
    if match is None:
        raise ValueError("Malformed checkpoint name")
    return int(match.group(1))


class StateCheckpointer:
    """
    Manages the lifecycle of distributed training checkpoints.

    This class handles saving and loading the training state (JobState object)
    using PyTorch Distributed Checkpoint (DCP). It manages checkpoint versioning,
    storage rotation (keeping only N latest), and synchronization across distributed ranks.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            stepper: Stepper,
            config: CheckpointingConfig,
            gc: ManualGarbageCollector,
            run_name: str | None
    ):
        """
        Constructs the StateCheckpoint object.

        Args:
            dist_context: The distributed context.
            stepper: The training stepper tracking the current iteration/step.
            config: Configuration object containing checkpointing parameters.
            gc: Garbage collector for manual memory management during IO.
            run_name: Optional specific run name to append to the save directory.
        """
        self._dist_context = dist_context
        self._stepper = stepper
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
        next_name = f"save-{self._stepper.current_step}"
        return self._save_dir / next_name

    def _purge_old_checkpoints(self):
        if not self._dist_context.is_main_process:
            return
        if not self._config.num_to_keep:
            return

        to_delete = self._get_sorted_checkpoint_dirs()[:-self._config.num_to_keep]

        for delete_dir in to_delete:
            self._dist_context.logger.info(f"Purging checkpoint {delete_dir}")
            shutil.rmtree(delete_dir)

    def _checkpoint(self, state: Stateful):
        next_checkpoint_id = self._next_checkpoint_id()

        self._dist_context.logger.info("Freeing up memory before checkpointing")
        self._free_memory()
        self._dist_context.logger.info("Waiting for world before saving checkpoint")
        self._dist_context.wait_world()
        self._dist_context.logger.info(f"Saving checkpoint {next_checkpoint_id}")

        save_from = {"state": state}
        dcp.save(
            state_dict=save_from,
            checkpoint_id=next_checkpoint_id
        )

        self._purge_old_checkpoints()
        self._free_memory()

        self._dist_context.logger.info("Waiting for world after saving checkpoint")
        self._dist_context.wait_world()
        self._dist_context.logger.info("Checkpoint successfully saved across the world")

    def checkpoint_if_needed(self, state: Stateful):
        """
        Checks if a checkpoint is due based on the configuration and saves if necessary.

        This checks the stepper to see if the current step matches the configured
        saving period (or if it is the final step).

        Args:
            state: The Stateful object to save.
        """

        if self._stepper.should_do_action(self._config.period_steps, enable_on_last_step_if_periodic=True):
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

        load_into = {
            "state": state
        }
        dcp.load(
            state_dict=load_into,
            checkpoint_id=last_checkpoint
        )
        self._free_memory()

        self._dist_context.logger.info("Waiting for world after loading checkpoint")
        self._dist_context.wait_world()
        self._dist_context.logger.info("Checkpoint successfully loaded across the world")

    def load_last_checkpoint(self, state: Stateful):
        """
        Attempts to load the most recent checkpoint available in the save directory.

        If no checkpoint is found, the state remains unchanged (starting from scratch).

        Args:
            state: The stateful object to which loaded parameters will be applied.
        """

        self._load(state)
