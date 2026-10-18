"""E0-T05 AC2: `key_for` is bitwise deterministic; `split_n`/`derive_seed` behave as documented."""

import subprocess
import sys

import jax
import numpy as np

from regimekit.utils.rng import derive_seed, key_for, split_n


def test_e0_t05_key_for_deterministic_in_process() -> None:
    """Calling `key_for` twice with the same arguments yields bitwise-identical keys."""
    a = jax.random.key_data(key_for(42, 1, 2))
    b = jax.random.key_data(key_for(42, 1, 2))
    np.testing.assert_array_equal(a, b)


def test_e0_t05_key_for_deterministic_across_processes() -> None:
    """`key_for(42, 1, 2)` is bitwise identical across two fresh processes."""
    code = (
        "import jax\n"
        "from regimekit.utils.rng import key_for\n"
        "print(list(jax.random.key_data(key_for(42, 1, 2))))\n"
    )
    results = [
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
        for _ in range(2)
    ]
    for result in results:
        assert result.returncode == 0, result.stderr
    assert results[0].stdout == results[1].stdout


def test_e0_t05_key_for_differs_by_ids() -> None:
    """Different purpose ids fold in differently."""
    a = jax.random.key_data(key_for(42, 1, 2))
    b = jax.random.key_data(key_for(42, 1, 3))
    assert not np.array_equal(a, b)


def test_e0_t05_split_n_returns_n_independent_keys() -> None:
    """`split_n` returns `n` keys, all distinct."""
    keys = split_n(key_for(0), 4)
    assert keys.shape == (4,)
    data = jax.random.key_data(keys)
    assert len({tuple(row.tolist()) for row in np.asarray(data)}) == 4


def test_e0_t05_derive_seed_deterministic_and_int() -> None:
    """`derive_seed` is a deterministic plain `int`, distinct for distinct inputs."""
    a = derive_seed(42, 1, 2)
    b = derive_seed(42, 1, 2)
    c = derive_seed(42, 1, 3)
    assert isinstance(a, int)
    assert a == b
    assert a != c
