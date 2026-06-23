import contextlib
from collections.abc import Iterator
from typing import Self

import torch
from torch import nn

from .layer import MoELayer


class RouterReplayRecorder:
    """Records the per-layer expert selection of every Mixture-of-Experts layer in a model.

    Installing the recorder onto a model binds every `MoELayer` to this recorder. While installed, each MoE
    layer that runs a non-replay forward hands its selected expert indices to this recorder, which keys them by the
    layer's fully-qualified module name. After the forward pass, `tape` returns a mapping from module name to the
    recorded selection, which can be fed back as the ``replay_indices`` model input to reproduce the exact same routing
    during training.

    The recorder owns the layer-to-name mapping; layers stay unaware of their identity within it. Layers may be called
    multiple times before `tape` is read (gradient accumulation, micro-batching): every call is appended and the
    captures of a layer are concatenated along the batch dimension in call order.

    This supports the case where d9d itself samples a rollout: the recorded routing is replayed in the subsequent
    training forward so the policy gradient is computed against the experts that actually generated the trajectory.
    When the rollout is produced by an external inference engine instead, the engine supplies an equivalent mapping
    directly and this recorder is not needed.
    """

    def __init__(self) -> None:
        """Constructs an empty, uninstalled recorder."""
        self._names: dict[MoELayer, str] = {}
        self._captures: dict[MoELayer, list[torch.Tensor]] = {}

    @contextlib.contextmanager
    def install(self, model: nn.Module) -> Iterator[Self]:
        """Binds every MoE layer of ``model`` to this recorder for the duration of the context.

        The recorded tape is keyed by each MoE layer's name relative to ``model`` (as reported by
        ``model.named_modules()``). Install the recorder on the same module whose ``forward`` will later consume the
        tape so the keys line up.

        Args:
            model: The model whose MoE layers should record their routing.

        Yields:
            This recorder, so that `tape` can be called within the context.

        Raises:
            RuntimeError: If the recorder is already installed.
        """
        if self._names:
            raise RuntimeError("RouterReplayRecorder is already installed.")

        for name, module in model.named_modules():
            if isinstance(module, MoELayer):
                module.bind_replay_recorder(self)
                self._names[module] = name

        try:
            yield self
        finally:
            for module in self._names:
                module.unbind_replay_recorder()
            self._names = {}
            self._captures = {}

    def capture(self, layer: MoELayer, expert_indices: torch.Tensor) -> None:
        """Stores the expert selection of a single MoE layer. Called by `MoELayer` during recording.

        Args:
            layer: The MoE layer that produced the selection.
            expert_indices: The selected expert indices. Shape: `(batch_size, seq_len, top_k)`.
        """
        self._captures.setdefault(layer, []).append(expert_indices.detach().clone())

    def tape(self) -> dict[str, torch.Tensor]:
        """Assembles the recorded selections into a mapping from module name to replay indices.

        Returns:
            A mapping from each MoE layer's module name to its recorded expert indices, with shape
            `(batch_size, seq_len, top_k)` (the per-call captures concatenated along the batch dimension). The mapping
            is ready to be passed back as the ``replay_indices`` model input.

        Raises:
            RuntimeError: If no routing has been recorded, or if some bound layers did not record (e.g. the forward
                pass supplied ``replay_indices`` and therefore skipped recording).
        """
        if not self._captures:
            raise RuntimeError("No routing was recorded; run a forward pass while the recorder is installed.")
        if self._captures.keys() != self._names.keys():
            raise RuntimeError(
                f"Expected all {len(self._names)} bound layers to record, but {len(self._captures)} did. "
                "Did the forward pass supply replay_indices (which skips recording)?"
            )
        return {self._names[layer]: torch.cat(chunks, dim=0) for layer, chunks in self._captures.items()}
