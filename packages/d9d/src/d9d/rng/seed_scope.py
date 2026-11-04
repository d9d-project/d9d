import hashlib


def _encode_key(key: str | int) -> bytes:
    # The type tag and the length prefix keep "1" apart from 1 and ("ab", "c") apart from ("a", "bc").
    match key:
        case str():
            tag, payload = b"s", key.encode()
        case _:
            tag, payload = b"i", str(key).encode()
    return tag + len(payload).to_bytes(8, "little") + payload


def _hash(*parts: bytes) -> int:
    # Python's hash() depends on PYTHONHASHSEED, so it would give other seeds in other processes.
    # 8 bytes give a 64-bit seed, the full range of torch.Generator.manual_seed().
    digest = hashlib.blake2b(b"".join(parts), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class SeedScope:
    """A node in the tree of seeds derived from the base seed of a job."""

    def __init__(self, seed: int):
        """Constructs the ``SeedScope`` object.

        Args:
            seed: The seed of the scope, in ``[0, 2**64)``.

        Raises:
            ValueError: If ``seed`` is out of range.
        """
        if not 0 <= seed < 2**64:
            raise ValueError(f"The seed ({seed}) must be in [0, 2**64).")
        self._seed = seed

    @property
    def seed(self) -> int:
        """The seed of this scope, in ``[0, 2**64)``. APIs that take a 32-bit seed must use it modulo ``2**32``."""
        return self._seed

    @staticmethod
    def root(base_seed: int) -> "SeedScope":
        """Returns the root seed scope of a job, with a seed hashed from the base seed.

        Args:
            base_seed: The base seed of the job. Any integer is valid.

        Returns:
            The root seed scope.
        """
        return SeedScope(_hash(b"root", _encode_key(base_seed)))

    def child(self, *keys: str | int) -> "SeedScope":
        """Returns the seed scope below this one, with a seed hashed from this seed and the keys.

        Args:
            *keys: The keys of the child, e.g. the name of a call and its counters.

        Returns:
            The child seed scope.
        """
        return SeedScope(_hash(self._seed.to_bytes(8, "little"), *map(_encode_key, keys)))
