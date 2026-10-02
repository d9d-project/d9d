import tarfile
import time
from contextlib import contextmanager
from pathlib import Path

import torch.profiler as tprof

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext


class Profiler:
    """Manages distributed performance profiling using PyTorch Profiler.

    This class wraps `torch.profiler` to provide automatic trace exporting,
    compression, and file naming consistent with the distributed DeviceMesh
    topology. It configures the schedule to repeat periodically based on
    the provided step counts.
    """

    def __init__(
        self,
        save_dir: Path,
        period_steps: int,
        warmup_steps: int,
        active_steps: int,
        record_shapes: bool,
        with_stack: bool,
        dist_context: DistributedContext,
    ):
        """Constructs a Profiler object.

        Args:
            save_dir: Directory where trace files will be saved.
            period_steps: Total length of a profiling cycle (wait + warmup + active).
            warmup_steps: Number of steps to ignore before recording to allow for warming-up.
            active_steps: Number of steps to actively record traces.
            record_shapes: Whether to record the input shapes of operators.
            with_stack: Whether to record the Python call stacks of operators.
            dist_context: The distributed context object.
        """
        self._save_dir = save_dir
        self._period = period_steps
        self._warmup = warmup_steps
        self._active = active_steps
        self._record_shapes = record_shapes
        self._with_stack = with_stack
        self._dist_context = dist_context

    def _get_save_file_name(self) -> str:
        if self._dist_context.mesh_params.is_distributed:
            mesh_regular = self._dist_context.mesh_for(REGULAR_DOMAIN)
            coord = mesh_regular.get_coordinate()
            if coord is None:
                raise RuntimeError("Invalid mesh")
            coord_str = "-".join(str(x) for x in coord)
            rank = mesh_regular.get_rank()
            return f"rank-{rank}-coord-{coord_str}-trace.json"
        else:
            return "trace.json"

    def _dump_trace(self, prof: tprof.profile):
        save_dir = self._save_dir / f"step_{prof.step_num}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / self._get_save_file_name()

        begin = time.monotonic()

        prof.export_chrome_trace(str(save_file))
        with tarfile.open(save_file.with_suffix(".tar.gz"), "w:gz") as tar:
            tar.add(save_file, arcname=save_file.name)
        save_file.unlink()

        end = time.monotonic()

        self._dist_context.logger.info(f"Finished dumping profiler traces in {end - begin:.2f} seconds")

    @contextmanager
    def open(self, start_step: int):
        """Opens a context manager for profiling execution.

        This sets up the `torch.profiler.profile` with a schedule derived from
        the initialization parameters. It captures both CPU and CUDA activities
        on all threads, and records operator shapes and stack traces if enabled.

        When the schedule triggers `on_trace_ready`, the trace is automatically
        exported to the `save_dir`, compressed into a `.tar.gz` file, and the
        raw JSON is removed to save space.

        Args:
            start_step: The current global step number to initialize the
                profiler state.

        Yields:
            The configured torch profiler instance.
        """
        wait = self._period - (self._active + self._warmup)
        warmup = self._warmup
        active = self._active

        with tprof.profile(
            activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA],
            schedule=tprof.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=self._dump_trace,
            record_shapes=self._record_shapes,
            with_stack=self._with_stack,
            # by default only the thread that enters the profiler is recorded, which misses background
            # threads such as data prefetching; torch exposes this option only via the private name
            experimental_config=tprof._ExperimentalConfig(profile_all_threads=True),  # noqa: SLF001
        ) as profiler:
            profiler.step_num = start_step
            yield profiler
