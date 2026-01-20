from llmadapt.lohotron.state.mapper.agg import ModelStateMapperParallel
from llmadapt.lohotron.state.mapper.base import ModelStateMapper
from llmadapt.lohotron.state.mapper.impl import ModelStateMapperIdentity
from torch import nn


def build_identity_mapper_for_saving(mapper: ModelStateMapper):
    mappers = []

    for state_group in mapper.state_dependency_groups():
        for output_name in state_group.outputs:
            mappers.append(ModelStateMapperIdentity(output_name))

    return ModelStateMapperParallel(mappers)


def build_model_state_mapper_from_module(model: nn.Module):
    return build_model_state_mapper_from_modules([model])


def build_model_state_mapper_from_modules(models: list[nn.Module]):
    return ModelStateMapperParallel(
        [ModelStateMapperIdentity(key) for model in models for key in model.state_dict()]
    )
