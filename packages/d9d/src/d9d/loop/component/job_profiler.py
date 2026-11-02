import tarfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import torch.profiler as tprof

from d9d.dist_context import REGULAR_DOMAIN, DistributedContext

from ..config import ProfilingConfig
from .job_schedule import JobSchedule


class JobProfiler:
    """Periodic profiler of the job loop based on ``torch.profiler``.

    The profiling cycle repeats every ``period_steps`` steps and follows the current step of the job schedule.
    Each trace is exported, compressed and named after the coordinates of the rank in the device mesh.
    """

    def __init__(self, dist_context: DistributedContext, config: ProfilingConfig | None, schedule: JobSchedule):
        """Constructs the ``JobProfiler`` object.

        Args:
            dist_context: The distributed context.
            config: The profiling configuration. ``None`` disables profiling.
            schedule: The schedule that tracks the current step of the loop.
        """
        self._dist_context = dist_context
        self._config = config
        self._schedule = schedule

    def _get_save_file_name(self) -> str:
        if self._dist_context.mesh_params.is_distributed:
            mesh_regular = self._dist_context.mesh_for(REGULAR_DOMAIN)
            coord = mesh_regular.get_coordinate()
            if coord is None:
                raise RuntimeError("The current rank is not part of the regular device mesh.")
            coord_str = "-".join(str(x) for x in coord)
            rank = mesh_regular.get_rank()
            return f"rank-{rank}-coord-{coord_str}-trace.json"
        else:
            return "trace.json"

    def _dump_trace(self, traces_dir: Path, prof: tprof.profile):
        save_dir = traces_dir / f"step_{prof.step_num}"
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
    def open(self) -> Generator[tprof.profile | None]:
        """Profiles the job loop in the block.

        The profiler records CPU and CUDA activity on all threads and counts steps from the current step of the
        schedule. Each finished trace is saved to ``traces_dir/step_<step>/`` as a ``.tar.gz`` archive.

        Yields:
            The configured ``torch.profiler.profile`` object if profiling is enabled, otherwise ``None``.
        """
        config = self._config
        if config is None:
            yield None
            return

        with tprof.profile(
            activities=[tprof.ProfilerActivity.CPU, tprof.ProfilerActivity.CUDA],
            schedule=tprof.schedule(
                wait=config.period_steps - (config.active_steps + config.warmup_steps),
                warmup=config.warmup_steps,
                active=config.active_steps,
            ),
            on_trace_ready=lambda prof: self._dump_trace(config.traces_dir, prof),
            record_shapes=config.record_shapes,
            with_stack=config.with_stack,
            # By default only the thread that enters the profiler is recorded, which misses background
            # threads such as data prefetching; torch exposes this option only under a private name.
            experimental_config=tprof._ExperimentalConfig(profile_all_threads=True),  # noqa: SLF001 - no public name
        ) as profiler:
            profiler.step_num = self._schedule.current_step
            yield profiler
