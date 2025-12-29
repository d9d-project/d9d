from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperIdentity


def identity_mapper_from_mapper_outputs(mapper: ModelStateMapper) -> ModelStateMapper:
    """
    Creates an identity mapper covering all outputs produced by the provided mapper.

    This function inspects the `state_dependency_groups()` of the input `mapper`,
    extracts every key listed in the `outputs` set of each group, and creates a
    corresponding `ModelStateMapperIdentity` for it.

    Args:
        mapper: The mapper whose output signature will be inspected to generate the new identity mapper.

    Returns:
        A composite mapper that acts as a pass-through for every key produced by the source `mapper`.
    """

    mappers = []

    for state_group in mapper.state_dependency_groups():
        for output_name in state_group.outputs:
            mappers.append(ModelStateMapperIdentity(output_name))

    return ModelStateMapperParallel(mappers)
