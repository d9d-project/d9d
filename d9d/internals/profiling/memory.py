import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch

from d9d.core.dist_context import DistributedContext

from ._artifact import archive_artifact, rank_artifact_path


class MemorySnapshotter:
    """Records CUDA caching allocator history and dumps it as distributed-aware memory snapshots.

    This class wraps `torch.cuda.memory._record_memory_history` and `torch.cuda.memory._snapshot`.
    Every recording starts with a cleared history, so a snapshot contains allocator events of its own
    recording only. `record_function` annotations are captured into the snapshot as well.

    Snapshots are serialized as JSON, compressed into `.tar.gz` archives and named consistently with the
    distributed DeviceMesh topology.
    """

    def __init__(self, save_dir: Path, max_entries: int, dist_context: DistributedContext):
        """Constructs a MemorySnapshotter object.

        Args:
            save_dir: Directory where snapshot files will be saved.
            max_entries: Maximum number of allocator events kept in the recorded history ring buffer.
            dist_context: The distributed context object.
        """
        self._save_dir = save_dir
        self._max_entries = max_entries
        self._dist_context = dist_context

    def start(self):
        """Starts recording the allocator history.

        Raises:
            RuntimeError: If the allocator history is already being recorded.
        """
        if torch._C._cuda_isHistoryEnabled():  # noqa: SLF001
            raise RuntimeError("CUDA memory history is already being recorded")

        torch.cuda.memory._record_memory_history(  # noqa: SLF001
            max_entries=self._max_entries, clear_history=True, global_record_annotations=True
        )

    def dump_and_stop(self, tag: str):
        """Takes a snapshot of the recorded history, stops recording and saves the snapshot.

        Args:
            tag: Name of the subdirectory of `save_dir` the snapshot is saved into.

        Raises:
            RuntimeError: If the allocator history is not being recorded.
        """
        if not torch._C._cuda_isHistoryEnabled():  # noqa: SLF001
            raise RuntimeError("CUDA memory history is not being recorded")

        snapshot = torch.cuda.memory._snapshot()  # noqa: SLF001
        torch.cuda.memory._record_memory_history(enabled=None)  # noqa: SLF001

        save_dir = self._save_dir / tag
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = rank_artifact_path(save_dir, self._dist_context, kind="memory")

        begin = time.monotonic()

        with save_file.open("w") as f:
            json.dump(snapshot, f)
        archive_artifact(save_file)

        end = time.monotonic()

        self._dist_context.logger.info(f"Finished dumping memory snapshot '{tag}' in {end - begin:.2f} seconds")

    @contextmanager
    def record(self, tag: str) -> Generator[None]:
        """Opens a context manager recording the allocator history for its whole scope.

        The snapshot is saved on exit even if the scope raised, so an out-of-memory
        failure leaves the snapshot of the allocations that caused it.

        Args:
            tag: Name of the subdirectory of `save_dir` the snapshot is saved into.
        """
        self.start()
        try:
            yield
        finally:
            self.dump_and_stop(tag)
