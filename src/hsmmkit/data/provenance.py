"""Content hashes for snapshot ids (E10-T04b): the clock- and domain-neutral part of the cache machinery."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def hash_bytes(data: bytes) -> str:
    """Hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    """Hex sha256 of the file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_arrays(names: Sequence[str], arrays: Sequence[np.ndarray]) -> str:
    """Hex sha256 over the names and the contiguous array buffers (dtype and shape included)."""
    digest = hashlib.sha256()
    for name, array in zip(names, arrays, strict=True):
        digest.update(name.encode("utf-8") + b"\0")
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.dtype).encode() + str(arr.shape).encode() + b"\0")
        digest.update(arr.tobytes())
    return digest.hexdigest()
