"""E1-T04 AC4: sampler moments within 3 SE of analytic, for n = 20000."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hsmmkit.model import dists

N = 20_000


def test_e1_t04_sample_normal_moments_within_3se() -> None:
    """Sample mean of `sample_normal` is within 3 SE of the analytic mean."""
    mean, sd = 0.3, 1.5
    xs = dists.sample_normal(jax.random.key(0), mean, sd, shape=(N,))
    se = sd / np.sqrt(N)
    assert abs(float(jnp.mean(xs)) - mean) <= 3 * se


def test_e1_t04_sample_student_t_moments_within_3se() -> None:
    """Sample mean of `sample_student_t` (nu > 2) is within 3 SE of the analytic mean."""
    loc, scale, nu = -0.2, 1.0, 8.0
    xs = dists.sample_student_t(jax.random.key(1), loc, scale, nu, shape=(N,))
    var = scale**2 * nu / (nu - 2.0)
    se = np.sqrt(var / N)
    assert abs(float(jnp.mean(xs)) - loc) <= 3 * se


def test_e1_t04_sample_categorical_frequencies_within_3se() -> None:
    """Empirical category frequencies from `sample_categorical` match `p` within 3 SE."""
    p = jnp.array([0.1, 0.2, 0.7])
    log_probs = jnp.log(p)
    keys = jax.random.split(jax.random.key(2), N)
    samples = jax.vmap(lambda k: dists.sample_categorical(k, log_probs))(keys)
    counts = jnp.bincount(samples, length=3)
    freqs = counts / N
    se = jnp.sqrt(p * (1 - p) / N)
    assert bool(jnp.all(jnp.abs(freqs - p) <= 3 * se))


def test_e1_t04_sample_pmf_frequencies_within_3se() -> None:
    """Empirical duration frequencies from `sample_pmf` match the pmf within 3 SE."""
    p = jnp.array([0.5, 0.3, 0.2])
    log_pmf = jnp.log(p)
    keys = jax.random.split(jax.random.key(3), N)
    samples = jax.vmap(lambda k: dists.sample_pmf(k, log_pmf))(keys)
    counts = jnp.bincount(samples, length=4)[1:]  # d in {1, 2, 3}
    freqs = counts / N
    se = jnp.sqrt(p * (1 - p) / N)
    assert bool(jnp.all(jnp.abs(freqs - p) <= 3 * se))


def test_e1_t04_samplers_are_jittable() -> None:
    """All samplers are jittable."""
    key = jax.random.key(4)
    assert jnp.isfinite(jax.jit(dists.sample_normal)(key, 0.0, 1.0))
    assert jnp.isfinite(jax.jit(lambda k: dists.sample_student_t(k, 0.0, 1.0, 6.0))(key))
    log_probs = jnp.log(jnp.array([0.3, 0.7]))
    assert jax.jit(lambda k: dists.sample_categorical(k, log_probs))(key) in (0, 1)
    log_pmf = jnp.log(jnp.array([0.5, 0.5]))
    assert jax.jit(lambda k: dists.sample_pmf(k, log_pmf))(key) in (1, 2)
