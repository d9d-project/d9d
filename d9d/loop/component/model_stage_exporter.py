from pathlib import Path

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.loop.control import ModelProvider, PrepareExportModelStageContext
from d9d.model_state.io import save_model_state_pipeline_parallel
from d9d.model_state.mapper.compose import ModelStateMapperParallel

from .model_stage_factory import TrackedModules


class ModelStageExporter:
    def __init__(self, model_provider: ModelProvider, modules: TrackedModules, dist_context: DistributedContext):
        self._model_provider = model_provider
        self._modules = modules
        self._dist_context = dist_context

    def export(self, save_dir: Path):
        mappers = []
        for stage in self._modules.modules:
            result = self._model_provider.prepare_export_model_stage(
                PrepareExportModelStageContext(model=stage, dist_context=self._dist_context)
            )
            mappers.append(result.state_mapper)
        save_model_state_pipeline_parallel(
            dest_dir=save_dir,
            mapper=ModelStateMapperParallel(mappers),
            device_mesh=self._dist_context.mesh_for(REGULAR_DOMAIN),
            pipeline_dim_name="pp",
            models=self._modules.modules,
            show_progress=True,
        )
