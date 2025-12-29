from d9d.model_state.mapper.abc import ModelStateMapper


def filter_empty_mappers(mappers: list[ModelStateMapper]) -> list[ModelStateMapper]:
    """
    Filters out mappers that have no effect (no inputs and no outputs).

    Args:
        mappers: The list of mappers to filter.

    Returns:
        A new list containing only active mappers.
    """
    result = []
    for mapper in mappers:
        for group in mapper.state_dependency_groups():
            if len(group.inputs) > 0 or len(group.outputs) > 0:
                result.append(mapper)
                break
    return result
