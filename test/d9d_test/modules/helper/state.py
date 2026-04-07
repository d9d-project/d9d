from collections.abc import Mapping
from typing import TypeAlias

import torch
from d9d.model_state.mapper import ModelStateMapper
from torch import nn
from torch.distributed.tensor import DTensor

from .compare import assert_angle_and_norm_close
from .tolerances import GradTolerance, grad_tolerance_for

GradToleranceOverrides: TypeAlias = GradTolerance | Mapping[str, GradTolerance]


def _apply_state_mapper(
    state_dict: dict[str, torch.Tensor],
    mapper: ModelStateMapper,
    keep_unmapped: bool = True,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = dict(state_dict) if keep_unmapped else {}

    produced: set[str] = set()
    for dependency_group in mapper.state_dependency_groups():
        missing = [key for key in dependency_group.inputs if key not in state_dict]
        if missing:
            raise KeyError(f"State mapper missing required keys: {missing}")

        group = {key: state_dict[key] for key in dependency_group.inputs}
        mapped = mapper.apply(group)

        missing_outputs = [key for key in dependency_group.outputs if key not in mapped]
        if missing_outputs:
            raise KeyError(f"State mapper did not produce required keys: {missing_outputs}")

        collisions = produced.intersection(mapped.keys())
        if collisions:
            raise KeyError(f"State mapper produced duplicate outputs: {sorted(collisions)}")

        produced.update(mapped.keys())
        out.update(mapped)

    return out


def clone_module_weights(from_module: nn.Module, to_module: nn.Module, map_with: ModelStateMapper) -> None:
    source_state = {name: value.detach().clone() for name, value in from_module.state_dict().items()}
    mapped = _apply_state_mapper(source_state, map_with, keep_unmapped=False)
    to_module.load_state_dict(mapped, strict=True)


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        return value.full_tensor()
    return value


def _module_grad_state_dict(
    module: nn.Module,
    none_as_zero: bool = True,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters():
        grad = param.grad
        if grad is None:
            if none_as_zero:
                out[name] = torch.zeros_like(_full_tensor(param.detach()))
        else:
            out[name] = _full_tensor(grad.detach()).clone()
    return out


def _grad_tolerance_for_state_key(
    state_key: str, state_dtype: torch.dtype, tolerances: GradToleranceOverrides | None
) -> GradTolerance:
    if tolerances is None:
        return grad_tolerance_for(state_dtype)
    if isinstance(tolerances, GradTolerance):
        return tolerances

    best_tol = grad_tolerance_for(state_dtype)
    best_match_len = -1
    for prefix, tol in tolerances.items():
        normalized_prefix = str(prefix).strip(".")
        if not normalized_prefix:
            continue
        if state_key == normalized_prefix or state_key.startswith(f"{normalized_prefix}."):
            match_len = len(normalized_prefix.split("."))
            if match_len > best_match_len:
                best_tol = tol
                best_match_len = match_len
    return best_tol


def _assert_grad_state_dict_close(
    actual: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor], tolerances: GradToleranceOverrides | None
) -> None:
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise AssertionError(f"gradient state_dict keys mismatch; missing={missing}, unexpected={unexpected}")

    for key in sorted(expected):
        tol = _grad_tolerance_for_state_key(key, expected[key].dtype, tolerances)
        assert_angle_and_norm_close(actual[key], expected[key], tol=tol, name=key)


def assert_mapped_gradients_close(
    from_module: nn.Module,
    to_module: nn.Module,
    map_with: ModelStateMapper,
    tolerances: GradToleranceOverrides | None = None,
    none_as_zero: bool = True,
) -> None:
    source_grads = _module_grad_state_dict(from_module, none_as_zero=none_as_zero)
    mapped_source_grads = _apply_state_mapper(source_grads, map_with, keep_unmapped=False)
    target_grads = _module_grad_state_dict(to_module, none_as_zero=none_as_zero)
    _assert_grad_state_dict_close(mapped_source_grads, target_grads, tolerances=tolerances)
