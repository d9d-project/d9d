from pathlib import Path

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.loop.control import ModelProvider, PrepareExportModelStageContext
from d9d.model_state.io import save_model_state, save_model_state_pipeline_parallel
from d9d.model_state.mapper.compose import ModelStateMapperParallel

from .model_stage_factory import TrackedModules


class ModelStageExporter:
    """Exports the model stages held by this process as a ``.safetensors`` checkpoint."""

    def __init__(self, model_provider: ModelProvider, modules: TrackedModules, dist_context: DistributedContext):
        """Constructs the ``ModelStageExporter`` object.

        Args:
            model_provider: The provider that maps each stage to its export state mapper.
            modules: The model stages held by this process.
            dist_context: The distributed context.
        """
        self._model_provider = model_provider
        self._modules = modules
        self._dist_context = dist_context

    def export(self, save_dir: Path):
        """Writes every stage this process holds to ``save_dir``.

        Args:
            save_dir: The directory to write the ``.safetensors`` shards and the index to.

        Raises:
            ValueError: If the run has no parallelism but holds more than one stage. Nothing then decides
                which stage writes the index.
        """
        alone = not self._dist_context.mesh_params.is_distributed
        stages = self._modules.modules
        if alone and len(stages) != 1:
            raise ValueError(f"The model stage count ({len(stages)}) must be 1 in a run without parallelism.")

        mappers = []
        for stage in stages:
            result = self._model_provider.prepare_export_model_stage(
                PrepareExportModelStageContext(model=stage, dist_context=self._dist_context)
            )
            mappers.append(result.state_mapper)
        mapper = ModelStateMapperParallel(mappers)

        if alone:
            save_model_state(dest_dir=save_dir, mapper=mapper, model=stages[0])
            return
        save_model_state_pipeline_parallel(
            dest_dir=save_dir,
            mapper=mapper,
            device_mesh=self._dist_context.mesh_for(REGULAR_DOMAIN),
            pipeline_dim_name="pp",
            models=stages,
            show_progress=True,
            position=self._dist_context.local_rank,
        )
