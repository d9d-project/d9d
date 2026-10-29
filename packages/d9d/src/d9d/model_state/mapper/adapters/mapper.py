from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperIdentity


def identity_mapper_from_mapper_outputs(mapper: ModelStateMapper) -> ModelStateMapper:
    """Creates an identity mapper for every output key of the given mapper.

    Args:
        mapper: The mapper whose ``state_dependency_groups()`` outputs are passed through.

    Returns:
        A composite mapper that passes through every key produced by ``mapper``.
    """
    mappers: list[ModelStateMapper] = []

    for state_group in mapper.state_dependency_groups():
        for output_name in state_group.outputs:
            mappers.append(ModelStateMapperIdentity(output_name))

    return ModelStateMapperParallel(mappers)
