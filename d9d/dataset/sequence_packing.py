import dataclasses
import random
from collections.abc import Iterator, Sequence
from typing import Any, Generic, Protocol, TypeVar, cast

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from d9d.core.pytree import tree_flatten, tree_leaves, tree_map, tree_unflatten
from d9d.core.types import TensorTree
from d9d.module.block.attention import SequencePacking

TSample = TypeVar("TSample", bound=TensorTree)

_T_co = TypeVar("_T_co", covariant=True)


class DatasetImplementingSampleLengthProtocol(Protocol[_T_co]):
    """Protocol for datasets that can report a sample's token count without materializing it.

    Distinct from ``DatasetImplementingSortKeyProtocol``: a sort key only has to *order* samples, so
    it may be approximate or offset, while ``sample_length`` is the exact token count that bin
    capacity is computed against.
    """

    def __len__(self) -> int:
        """Returns the total number of items in the dataset."""
        ...

    def sample_length(self, index: int) -> int:
        """Returns the token count of the item at the given index.

        Must equal the length of every tensor leaf of ``self[index]`` along dim 0.

        Args:
            index: The index of the item.

        Returns:
            The item's exact token count.
        """
        ...

    def __getitem__(self, index: int) -> _T_co:
        """Retrieves the item at the specific index."""
        ...


@dataclasses.dataclass
class PackedSequence(Generic[TSample]):
    """A single packed row: the concatenated sample tokens plus their block-diagonal segment descriptor.

    Keeping the tokens and the packing descriptor as separate members - rather than smuggling
    ``cu_seqlens`` into the sample under a reserved key - keeps the two roles distinct and lets a task
    lift ``packing`` straight into a ``SequenceShared`` without name collisions.

    ``tokens`` preserves the base sample's pytree structure (dict, dataclass, nested, ...); every leaf
    is the corresponding per-segment tensor concatenated across the packed samples.

    Attributes:
        tokens: The concatenated tokens, structurally identical to a single base sample.
        packing: The block-diagonal segment descriptor for the row.
    """

    tokens: TSample
    packing: SequencePacking


def _sample_length(sample: TensorTree) -> int:
    """Returns the shared length of every leaf in a sample, validating they agree.

    Returns:
        The length of the sample's tensors along dim 0.

    Raises:
        ValueError: If the sample has no tensors, or its tensors differ in length.
    """
    lengths = {leaf.shape[0] for leaf in tree_leaves(sample)}
    if len(lengths) != 1:
        raise ValueError(f"All tensors in a sample must share the same length, got lengths {lengths}")
    return next(iter(lengths))


def pack_samples(samples: Sequence[TSample]) -> PackedSequence[TSample]:
    """Concatenates several samples into a single packed row with a block-diagonal segment descriptor.

    Every sample is a pytree of 1-D tensors that share a length (the sample's token count). Samples
    share a structure; corresponding leaves are concatenated across samples, so a leaf that restarts
    per sample (e.g. ``position_ids`` built as ``arange(len)``) naturally restarts per packed segment.
    The per-segment boundaries are recorded in the returned ``SequencePacking`` so a downstream
    variable-length attention backend can attend block-diagonally.

    Args:
        samples: The samples to pack, each a pytree of equal-length 1-D tensors sharing one structure.

    Returns:
        The packed row: concatenated tokens (same structure as a single sample) plus their descriptor.

    Raises:
        ValueError: If ``samples`` is empty, samples disagree on their structure, or a sample's
            tensors differ in length.
    """
    if len(samples) == 0:
        raise ValueError("Cannot pack an empty sequence of samples")

    lengths = [_sample_length(sample) for sample in samples]

    # Flatten every sample against the first sample's structure; a mismatch raises here.
    per_sample_leaves, treespec = tree_flatten(samples[0])
    all_leaves = [per_sample_leaves]
    for sample in samples[1:]:
        leaves, spec = tree_flatten(sample)
        if spec != treespec:
            raise ValueError(f"All samples must share the same structure, got {spec} and {treespec}")
        all_leaves.append(leaves)

    concatenated = [torch.cat(leaves, dim=0) for leaves in zip(*all_leaves, strict=True)]
    tokens = cast(TSample, tree_unflatten(treespec, concatenated))

    cu_seqlens = torch.zeros(len(samples) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)

    return PackedSequence(tokens=tokens, packing=SequencePacking(cu_seqlens=cu_seqlens, max_seqlen=max(lengths)))


def _greedy_bins(lengths: Sequence[int], max_length: int) -> list[list[int]]:
    bins: list[list[int]] = []
    current: list[int] = []
    current_length = 0
    for index, length in enumerate(lengths):
        if length > max_length:
            raise ValueError(f"Sample {index} has length {length} exceeding max_length {max_length}")
        if current_length + length > max_length:
            bins.append(current)
            current = []
            current_length = 0
        current.append(index)
        current_length += length
    if current:
        bins.append(current)
    return bins


class SequencePackingDataset(Dataset[PackedSequence[TSample]], Stateful, Generic[TSample]):
    """A sized dataset that packs consecutive samples into fixed-capacity rows, precomputed up front.

    The bin assignment - which samples make up each packed row - is fixed for the dataset's lifetime,
    so the dataset is a plain ``index -> packed row`` map with a known length.
    """

    def __init__(
        self,
        base_dataset: DatasetImplementingSampleLengthProtocol[TSample],
        bins: list[list[int]],
    ):
        """Constructs a SequencePackingDataset object over an already-computed bin assignment.

        Args:
            base_dataset: The underlying dataset, indexed by the entries of ``bins``.
            bins: The packed rows, each a list of the base-dataset indices it concatenates, in order.
                Assumed to be a valid assignment - see :meth:`build` to compute one.
        """
        self._base_dataset = base_dataset
        self._bins = bins

    @classmethod
    def build(
        cls,
        base_dataset: DatasetImplementingSampleLengthProtocol[TSample],
        max_length: int,
        init_seed: int | None = None,
        show_progress: bool = True,
        position: int | None = None,
    ) -> "SequencePackingDataset[TSample]":
        """Builds a packing dataset by computing the bin assignment from the base samples' lengths.

        Reads every sample's length, length-sorts them so that similar lengths share a row (tight
        packing), greedily groups them into rows of at most ``max_length`` tokens, then shuffles the
        rows.

        Args:
            base_dataset: The underlying dataset. Its ``sample_length`` must return the sample's exact
                token count.
            max_length: The maximum number of tokens in a packed row.
            init_seed: Seed for shuffling the packed rows.
            show_progress: Whether to display a progress bar over the length pass.
            position: Row index for the tqdm bar. Pass the process local rank to stack one bar
                per rank without interleaving. ``None`` lets tqdm use its default (single bar).

        Returns:
            A dataset over the computed packed rows.

        Raises:
            ValueError: If ``max_length`` is not positive, or a sample is longer than ``max_length``.
        """
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        desc = f"Packing Sequences [{position}]" if position is not None else "Packing Sequences"
        lengths = [
            base_dataset.sample_length(index)
            for index in tqdm(
                range(len(base_dataset)),
                desc=desc,
                disable=not show_progress,
                position=position,
                leave=True,
            )
        ]
        order = sorted(range(len(lengths)), key=lambda index: lengths[index])
        local_bins = _greedy_bins([lengths[index] for index in order], max_length)

        bins = [[order[local] for local in bin_locals] for bin_locals in local_bins]
        random.Random(init_seed).shuffle(bins)

        return cls(base_dataset=base_dataset, bins=bins)

    def __getitem__(self, index: int) -> PackedSequence[TSample]:
        """Retrieves the packed row at the given index.

        Args:
            index: The packed-row index.

        Returns:
            The packed row (see :func:`pack_samples`).
        """
        return pack_samples([self._base_dataset[sample_index] for sample_index in self._bins[index]])

    def __len__(self) -> int:
        """Returns the number of packed rows.

        Returns:
            The packed-row count.
        """
        return len(self._bins)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {"bins": self._bins}
        if isinstance(self._base_dataset, Stateful):
            state["base_dataset"] = self._base_dataset.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._bins = state_dict["bins"]
        if isinstance(self._base_dataset, Stateful):
            self._base_dataset.load_state_dict(state_dict["base_dataset"])


class StreamingSequencePackingDataset(IterableDataset[PackedSequence[TSample]], Stateful, Generic[TSample]):
    """An unsized iterable dataset that packs samples into fixed-capacity rows on the fly.

    Consumes a stateful iterable of samples and greedily emits a packed row whenever the next sample
    would overflow ``max_length`` tokens, flushing any partial row at the end of the stream. Because
    the number of rows is data-dependent, the dataset has no length; a downstream job must take its
    duration from configuration rather than from the data.

    The in-flight buffer and the base iterable's position are checkpointed, so iteration resumes
    exactly from a saved state.
    """

    def __init__(
        self,
        base_dataset: IterableDataset[TSample],
        max_length: int,
    ):
        """Constructs a StreamingSequencePackingDataset object.

        Args:
            base_dataset: The underlying stateful iterable of samples.
            max_length: The maximum number of tokens in a packed row.

        Raises:
            ValueError: If ``max_length`` is not positive, or the base dataset is not ``Stateful``.
        """
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if not isinstance(base_dataset, Stateful):
            raise ValueError("StreamingSequencePackingDataset requires a Stateful base dataset")

        self._base_dataset = base_dataset
        self._max_length = max_length
        self._buffer: list[TSample] = []

    def __iter__(self) -> Iterator[PackedSequence[TSample]]:
        """Iterates the base dataset, emitting a packed row whenever the buffer is full.

        Yields:
            One packed row (see :func:`pack_samples`) at a time.

        Raises:
            ValueError: If a single sample is longer than ``max_length``.
        """
        buffer_length = sum(_sample_length(sample) for sample in self._buffer)
        for sample in self._base_dataset:
            length = _sample_length(sample)
            if length > self._max_length:
                raise ValueError(f"Sample has length {length} exceeding max_length {self._max_length}")
            if buffer_length + length > self._max_length:
                # Start the new buffer with the triggering sample *before* yielding, so a checkpoint
                # taken while suspended at this yield holds exactly the not-yet-emitted samples.
                packed = pack_samples(self._buffer)
                self._buffer = [sample]
                buffer_length = length
                yield packed
            else:
                self._buffer.append(sample)
                buffer_length += length

        if self._buffer:
            packed = pack_samples(self._buffer)
            self._buffer = []
            yield packed

    def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": self._buffer,
            "base_dataset": self._base_dataset.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._buffer = state_dict["buffer"]
        self._base_dataset.load_state_dict(state_dict["base_dataset"])


def pack_collate(samples: Sequence[PackedSequence[TSample]]) -> PackedSequence[TSample]:
    """Collates a single packed row into a batch-of-one microbatch.

    A packed row is already one microbatch, so the loader must use ``batch_size=1``. Every token leaf
    gains a leading batch dimension of size one (giving the ``(1, total)`` layout the model expects),
    while the segment descriptor is passed through unchanged, since it describes that single row.

    Args:
        samples: A one-element sequence holding the packed row.

    Returns:
        The packed row with token leaves unsqueezed to a leading batch dimension of one.

    Raises:
        ValueError: If ``samples`` does not hold exactly one packed row.
    """
    if len(samples) != 1:
        raise ValueError(f"pack_collate expects exactly one packed row (batch_size=1), got {len(samples)}")

    row = samples[0]
    tokens = cast(TSample, tree_map(lambda leaf: leaf.unsqueeze(0), row.tokens))
    return PackedSequence(tokens=tokens, packing=row.packing)
