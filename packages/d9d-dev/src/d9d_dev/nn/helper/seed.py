from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def torch_seed(seed: int) -> Iterator[None]:
    # torch.random.fork_rng() always forks CPU RNG as well, so CPU-only random
    # ops inside this context stay deterministic together with the current CUDA device.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed(int(seed))
        yield
