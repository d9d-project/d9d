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
    layer's fully-qualified module name. After the forward pass, `tape` returns the recorded selection, which
    can be fed back as the ``replay_indices`` model input to reproduce the exact same routing during training.

    The recorder owns the layer-to-name mapping; layers stay unaware of their identity within it.

    Each forward over a micro-batch is recorded as one entry of the tape. A step that runs several micro-batches
    (gradient accumulation, sequence packing, dynamic token budgets) therefore produces a list of per-micro-batch
    mappings, one per micro-batch, in call order. The micro-batches are deliberately NOT merged into a single tensor,
    as their sequence lengths may differ. ``tape()[i]`` is the replay mapping for the i-th micro-batch and is
    fed to that micro-batch's forward.

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

    def tape(self) -> list[dict[str, torch.Tensor]]:
        """Assembles the recorded selections into one replay mapping per micro-batch.

        Returns:
            A list with one entry per recorded micro-batch, in call order. Each entry maps a MoE layer's module name
            to that micro-batch's recorded expert indices, of shape `(batch_size, seq_len, top_k)`. Entry ``i`` is the
            ``replay_indices`` mapping for the i-th micro-batch. Micro-batches are kept separate (not concatenated), so
            their sequence lengths may differ.

        Raises:
            RuntimeError: If no routing has been recorded; if some bound layers did not record at all (e.g. the
                forward pass supplied ``replay_indices`` and therefore skipped recording); or if the bound layers
                recorded a different number of micro-batches (each MoE layer must run exactly once per micro-batch).
        """
        if not self._captures:
            raise RuntimeError("No routing was recorded; run a forward pass while the recorder is installed.")
        if self._captures.keys() != self._names.keys():
            raise RuntimeError(
                f"Expected all {len(self._names)} bound layers to record, but {len(self._captures)} did. "
                "Did the forward pass supply replay_indices (which skips recording)?"
            )

        microbatch_counts = {len(chunks) for chunks in self._captures.values()}
        if len(microbatch_counts) != 1:
            raise RuntimeError(
                f"MoE layers recorded a different number of micro-batches ({sorted(microbatch_counts)}); "
                "each layer must run exactly once per micro-batch."
            )
        num_microbatches = microbatch_counts.pop()

        return [
            {self._names[layer]: chunks[microbatch] for layer, chunks in self._captures.items()}
            for microbatch in range(num_microbatches)
        ]
