import dataclasses
import threading
import time

import pytest
import torch
from d9d.dataset import FixedCountMicrobatchPacker, PinMemoryMicrobatchPackStream
from d9d.loop.component import DirectDevicePackStream, PrefetchingDevicePackStream, build_device_pack_stream
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

_NUM_SAMPLES = 24
_MICROBATCH_SIZE = 2


@dataclasses.dataclass
class _Microbatch:
    ids: torch.Tensor
    meta: dict[str, torch.Tensor]
    size: int


class _Dataset(Dataset):
    def __len__(self):
        return _NUM_SAMPLES

    def __getitem__(self, index):
        return index


def _collate(indices):
    ids = torch.tensor(indices)
    return _Microbatch(ids=ids, meta={"double": ids * 2}, size=len(indices))


def _make_stream(num_workers=0, pin_memory=True):
    loader = StatefulDataLoader(
        _Dataset(), batch_size=_MICROBATCH_SIZE, collate_fn=_collate, num_workers=num_workers, drop_last=True
    )
    stream = FixedCountMicrobatchPacker(loader, microbatches_per_step=2)
    return PinMemoryMicrobatchPackStream(stream) if pin_memory else stream


def _contents(pack):
    return [microbatch.ids.tolist() for microbatch in pack]


def _consume(device_stream, num_steps=None):
    seen = []
    for pack in device_stream:
        seen.append(_contents(pack))
        if len(seen) == num_steps:
            break
    return seen


@pytest.fixture(scope="module")
def reference():
    return [_contents(pack) for pack in _make_stream(pin_memory=False)]


@pytest.mark.local
@pytest.mark.parametrize("prefetch_factor", [0, 1, 3])
@pytest.mark.parametrize("pin_memory", [True, False])
def test_hands_out_packs_on_device_in_order(reference, prefetch_factor, pin_memory):
    device_stream = build_device_pack_stream(_make_stream(pin_memory=pin_memory), "cuda", prefetch_factor)

    seen = []
    for pack in device_stream:
        for microbatch in pack:
            assert isinstance(microbatch, _Microbatch)
            assert microbatch.ids.is_cuda
            assert microbatch.meta["double"].is_cuda
            assert torch.equal(microbatch.meta["double"], microbatch.ids * 2)
            assert microbatch.size == _MICROBATCH_SIZE
        seen.append(_contents(pack))

    assert seen == reference


@pytest.mark.local
@pytest.mark.parametrize("prefetch_factor", [1, 3])
def test_handed_out_pack_waits_for_its_copy(prefetch_factor):
    numel = 16 * 1024 * 1024
    # distinct per parametrization, so device memory reused from the other run cannot hold the result by chance
    values = [1000 * prefetch_factor + step for step in range(3)]

    class _LargeStream:
        total_steps = len(values)

        def __iter__(self):
            for value in values:
                yield [torch.full((numel,), value, dtype=torch.int32).pin_memory()]

        def state_dict(self):
            return {}

        def load_state_dict(self, state_dict):
            pass

    device_stream = build_device_pack_stream(_LargeStream(), "cuda", prefetch_factor)
    # hold the copies back, so reading a pack without waiting for its copy would see unfinished data
    with torch.cuda.stream(device_stream._copy_stream):
        torch.cuda._sleep(1_000_000_000)

    sums = [pack[0].sum(dtype=torch.int64) for pack in device_stream]

    assert [s.item() for s in sums] == [numel * value for value in values]


@pytest.mark.local
@pytest.mark.parametrize("steps_before_checkpoint", [1, 3])
@pytest.mark.parametrize(
    ("save_prefetch_factor", "load_prefetch_factor", "num_workers"),
    [
        (0, 0, 0),
        (1, 1, 0),
        (3, 3, 0),
        (3, 0, 0),
        (0, 2, 0),
        # worker loaders hand out their state through snapshots sent from the workers
        (3, 3, 2),
    ],
)
def test_resume_from_checkpoint_matches_uninterrupted_run(
    reference, steps_before_checkpoint, save_prefetch_factor, load_prefetch_factor, num_workers
):
    device_stream = build_device_pack_stream(_make_stream(num_workers=num_workers), "cuda", save_prefetch_factor)
    iterator = iter(device_stream)
    before = [_contents(next(iterator)) for _ in range(steps_before_checkpoint)]
    # the checkpoint is taken at the step boundary, while later packs are already prefetched
    state = device_stream.state_dict()

    resumed = build_device_pack_stream(_make_stream(num_workers=num_workers), "cuda", load_prefetch_factor)
    resumed.load_state_dict(state)

    assert before + _consume(resumed) == reference


@pytest.mark.local
@pytest.mark.parametrize("prefetch_factor", [0, 1, 3])
def test_iterating_again_behaves_like_iterating_the_stream_again(prefetch_factor):
    plain = _make_stream()
    plain_first = [_contents(pack) for _, pack in zip(range(2), plain, strict=False)]
    plain_second = [_contents(pack) for pack in plain]

    device_stream = build_device_pack_stream(_make_stream(), "cuda", prefetch_factor)

    assert _consume(device_stream, num_steps=2) == plain_first
    assert _consume(device_stream) == plain_second


@pytest.mark.local
def test_builder_picks_the_stream_by_prefetch_factor():
    assert isinstance(build_device_pack_stream(_make_stream(), "cuda", 0), DirectDevicePackStream)
    assert isinstance(build_device_pack_stream(_make_stream(), "cuda", 2), PrefetchingDevicePackStream)

    with pytest.raises(ValueError, match="non-negative"):
        build_device_pack_stream(_make_stream(), "cuda", -1)
    with pytest.raises(ValueError, match="positive"):
        PrefetchingDevicePackStream(_make_stream(), "cuda", 0)


class _RecordingStream:
    """Yields single-tensor packs and records which thread pulls them."""

    def __init__(self, num_packs, fail_at=None):
        self.total_steps = num_packs
        self.pulled = 0
        self.pulling_threads = set()
        self._fail_at = fail_at

    def __iter__(self):
        for step in range(self.total_steps):
            if step == self._fail_at:
                raise RuntimeError("broken stream")
            self.pulled += 1
            self.pulling_threads.add(threading.current_thread())
            yield [torch.full((4,), step)]

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def _prefetch_threads():
    return [thread for thread in threading.enumerate() if thread.name == "d9d-pack-prefetch"]


def _wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


@pytest.mark.local
@pytest.mark.parametrize("prefetch_factor", [1, 3])
def test_stream_is_pulled_off_the_main_thread_at_most_prefetch_factor_ahead(prefetch_factor):
    stream = _RecordingStream(num_packs=10)
    iterator = iter(build_device_pack_stream(stream, "cuda", prefetch_factor))

    next(iterator)

    # the handed out pack plus up to prefetch_factor packs ahead, but no more
    assert _wait_until(lambda: stream.pulled == 1 + prefetch_factor)
    time.sleep(0.1)
    assert stream.pulled == 1 + prefetch_factor
    assert threading.main_thread() not in stream.pulling_threads


@pytest.mark.local
def test_stream_error_reaches_the_loop():
    iterator = iter(build_device_pack_stream(_RecordingStream(num_packs=5, fail_at=2), "cuda", 1))

    assert next(iterator)[0].tolist() == [0, 0, 0, 0]
    assert next(iterator)[0].tolist() == [1, 1, 1, 1]
    with pytest.raises(RuntimeError, match="broken stream"):
        next(iterator)


@pytest.mark.local
def test_leaving_the_iteration_stops_the_background_thread():
    stream = _RecordingStream(num_packs=10)
    iterator = iter(build_device_pack_stream(stream, "cuda", 2))
    next(iterator)
    # the producer is now blocked waiting for a free slot
    assert _wait_until(lambda: stream.pulled == 3)
    assert len(_prefetch_threads()) == 1

    # close from a helper thread, so a producer that never stops fails the test instead of hanging it
    closer = threading.Thread(target=iterator.close, daemon=True)
    closer.start()
    closer.join(timeout=5)

    assert not closer.is_alive(), "the background thread did not stop"
    assert _prefetch_threads() == []
