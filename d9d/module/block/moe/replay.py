import contextlib
from collections.abc import Iterator
from typing import Self

import torch
from torch import nn

from .layer import MoELayer


class RouterReplayRecorder:
    """Records the per-layer expert selection of every Mixture-of-Experts layer in a model.

    Installing the recorder onto a model binds every `MoELayer` to this recorder and assigns it a stable
    integer id (the order in which the layers appear in ``model.modules()``). While installed, each MoE layer that
    runs a non-replay forward writes its selected expert indices into this recorder. After the forward pass,
    `tape` assembles those per-layer selections into a single tensor that can be fed back as the
    ``replay_indices`` model input to reproduce the exact same routing during training.

    This supports the case where d9d itself samples a rollout: the recorded routing is replayed in the subsequent
    training forward so the policy gradient is computed against the experts that actually generated the trajectory.
    When the rollout is produced by an external inference engine instead, the engine supplies an equivalent tensor
    directly and this recorder is not needed.
    """

    def __init__(self) -> None:
        """Constructs an empty, uninstalled recorder."""
        self._layers: list[MoELayer] = []
        self._buffer: dict[int, torch.Tensor] = {}

    @contextlib.contextmanager
    def install(self, model: nn.Module) -> Iterator[Self]:
        """Binds every MoE layer of ``model`` to this recorder for the duration of the context.

        Args:
            model: The model whose MoE layers should record their routing.

        Yields:
            This recorder, so that `tape` can be called within the context.

        Raises:
            RuntimeError: If the recorder is already installed.
        """
        if self._layers:
            raise RuntimeError("RouterReplayRecorder is already installed.")

        for layer_id, layer in enumerate(module for module in model.modules() if isinstance(module, MoELayer)):
            layer.bind_replay_recorder(self, layer_id)
            self._layers.append(layer)

        try:
            yield self
        finally:
            for layer in self._layers:
                layer.unbind_replay_recorder()
            self._layers = []
            self._buffer = {}

    def capture(self, layer_id: int, expert_indices: torch.Tensor) -> None:
        """Stores the expert selection of a single MoE layer. Called by `MoELayer` during recording.

        Args:
            layer_id: Stable identifier of the layer that produced the selection.
            expert_indices: The selected expert indices. Shape: `(batch_size, seq_len, top_k)`.
        """
        self._buffer[layer_id] = expert_indices.detach().clone()

    def tape(self) -> torch.Tensor:
        """Assembles the recorded selections of all layers into a single replay tensor.

        Returns:
            The recorded expert indices for every MoE layer. Shape: `(batch_size, num_layers, seq_len, top_k)`,
            ready to be passed back as the ``replay_indices`` model input.

        Raises:
            RuntimeError: If no routing has been recorded, or if some layers did not record (e.g. the forward pass
                supplied ``replay_indices`` and therefore skipped recording).
        """
        if not self._buffer:
            raise RuntimeError("No routing was recorded; run a forward pass while the recorder is installed.")
        if len(self._buffer) != len(self._layers):
            raise RuntimeError(
                f"Expected {len(self._layers)} layers to record, but {len(self._buffer)} did. "
                "Did the forward pass supply replay_indices (which skips recording)?"
            )
        return torch.stack([self._buffer[layer_id] for layer_id in range(len(self._layers))], dim=1)
