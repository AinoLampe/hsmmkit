"""E3-T03: `DurationFamily` -- geometric/negbin advance_prob, log_pmf, mean_sojourn."""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.model.duration.geometric import GeometricDuration
from regimekit.model.duration.negbin import NegBinDuration

GEO = GeometricDuration()


def _theta(rate: list[float]) -> dict[str, jnp.ndarray]:
    return {"duration.rate": jnp.asarray(rate)}


# --- AC1: phase-chain simulation (r=3, a=0.05): mean within 3 SE of r/a, min == r ---------------


def test_e3_t03_phase_chain_simulation_mean_and_minimum() -> None:
    """Simulating the r=3 phase chain: empirical mean sojourn within 3 SE of r/a; min == r."""
    rng = np.random.default_rng(0)
    r, a, n_sim = 3, 0.05, 20000

    durations = np.empty(n_sim, dtype=np.int64)
    for k in range(n_sim):
        phase, d = 1, 0
        while phase <= r:
            d += 1
            if rng.uniform() < a:
                phase += 1
        durations[k] = d

    mean_d = r / a
    var_d = r * (1.0 - a) / a**2
    se = np.sqrt(var_d / n_sim)
    assert abs(durations.mean() - mean_d) < 3 * se
    assert durations.min() == r


# --- AC2: negbin r=1 equals geometric ------------------------------------------------------------


def test_e3_t03_negbin_r1_equals_geometric_advance_prob_bitwise() -> None:
    """`negbin` with `r=1` equals `geometric`'s `advance_prob`, bitwise."""
    nb = NegBinDuration(r=(1, 1, 1))
    theta = _theta([0.05, 0.1, 0.2])
    dt = jnp.full((10,), 1.0 / 252.0)
    np.testing.assert_array_equal(
        np.asarray(nb.advance_prob(theta, dt, None)), np.asarray(GEO.advance_prob(theta, dt, None))
    )


def test_e3_t03_negbin_r1_equals_geometric_log_pmf() -> None:
    """`negbin` with `r=1` equals `geometric`'s `log_pmf`, within 1e-12."""
    nb = NegBinDuration(r=(1, 1, 1))
    theta = _theta([0.05, 0.1, 0.2])
    out_nb = np.asarray(nb.log_pmf(theta, d_max=50, nominal_dt=1.0 / 252.0, z_entry=None))
    out_geo = np.asarray(GEO.log_pmf(theta, d_max=50, nominal_dt=1.0 / 252.0, z_entry=None))
    np.testing.assert_allclose(out_nb, out_geo, atol=1e-12)


# --- AC3: irregular dt --------------------------------------------------------------------------


def test_e3_t03_irregular_dt_matches_formula_per_row() -> None:
    """`advance_prob` shape `(T, N)`, equals `1 - exp(-lambda * dt_t)` per row for irregular `dt`."""
    theta = _theta([0.05, 0.2])
    dt = jnp.asarray([1.0 / 252.0, 3.0 / 252.0, 1.0 / 52.0])
    out = np.asarray(GEO.advance_prob(theta, dt, None))
    assert out.shape == (3, 2)
    rate = np.array([0.05, 0.2])
    expected = 1.0 - np.exp(-rate[None, :] * np.asarray(dt)[:, None])
    np.testing.assert_allclose(out, expected, atol=1e-12)


# --- AC4: covariate link -------------------------------------------------------------------------


def test_e3_t03_zero_coefficients_reproduce_constant_rate() -> None:
    """`c = 0` reproduces the constant-`lambda` advance probability."""
    family = GeometricDuration(cov_idx=(0,))
    theta_plain = _theta([0.05])
    theta_cov = {**theta_plain, "duration.c": jnp.zeros((1, 1))}
    dt = jnp.full((5,), 1.0 / 252.0)
    z = jnp.asarray(np.random.default_rng(1).normal(size=(5, 1)))

    plain = family.advance_prob(theta_plain, dt, None)
    zero_cov = family.advance_prob(theta_cov, dt, z)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(zero_cov), atol=1e-12)


def test_e3_t03_positive_c_and_z_increases_advance_prob_monotonic() -> None:
    """Positive `c` with positive `z` increases `a_i(t)` (monotonic in `z`)."""
    family = GeometricDuration(cov_idx=(0,))
    theta = {**_theta([0.05]), "duration.c": jnp.array([[0.5]])}
    dt = jnp.full((1,), 1.0 / 252.0)

    zs = [0.0, 0.5, 1.0, 2.0]
    vals = [float(family.advance_prob(theta, dt, jnp.array([[z]]))[0, 0]) for z in zs]
    assert all(b > a for a, b in itertools.pairwise(vals))


# --- AC5: Appendix E (4 dp) -----------------------------------------------------------------------


@pytest.mark.parametrize(("rate", "expected_a"), [(2.0, 0.0079), (6.0, 0.0235)])
def test_e3_t03_appendix_e_daily_advance_prob(rate: float, expected_a: float) -> None:
    """`lambda=2/yr` daily -> `a=0.0079`; `lambda=6/yr` -> `0.0235` (4 dp)."""
    theta = _theta([rate])
    dt = jnp.full((1,), 1.0 / 252.0)
    a = float(GEO.advance_prob(theta, dt, None)[0, 0])
    assert round(a, 4) == pytest.approx(expected_a)


# --- AC6: steps clock -----------------------------------------------------------------------------


def test_e3_t03_steps_clock_advance_prob_and_mean_sojourn() -> None:
    """Steps clock: `dt=ones(T)`, `lambda=0.05/step` -> `a=1-e^-0.05`; `mean_sojourn(theta,1.0)==r/a`."""
    nb = NegBinDuration(r=(3,))
    theta = _theta([0.05])
    dt = jnp.ones((4,))
    a = np.asarray(nb.advance_prob(theta, dt, None))
    np.testing.assert_allclose(a, 1.0 - np.exp(-0.05), atol=1e-12)

    mean_sojourn = float(nb.mean_sojourn(theta, 1.0)[0])
    expected = 3 / (1.0 - np.exp(-0.05))
    assert mean_sojourn == pytest.approx(expected, abs=1e-9)


def test_e3_t03_init_mean_sojourn_steps_geometric_bitwise() -> None:
    """`init_mean_sojourn_steps=[20]` with `r=1` yields `lambda=0.05` (bitwise, `nominal_dt=1`)."""
    theta0 = GEO.init_from_mean_sojourn(jnp.asarray([20.0]), nominal_dt=1.0)
    assert float(theta0["duration.rate"][0]) == 0.05


def test_e3_t03_init_mean_sojourn_steps_negbin() -> None:
    """`NegBinDuration.init_from_mean_sojourn` uses `r_i / (mean_steps_i * nominal_dt)`."""
    nb = NegBinDuration(r=(3,))
    theta0 = nb.init_from_mean_sojourn(jnp.asarray([60.0]), nominal_dt=1.0)
    assert float(theta0["duration.rate"][0]) == pytest.approx(3.0 / 60.0)


# --- covariate-adjusted log_pmf/mean_sojourn (z_entry path) ---------------------------------------


def test_e3_t03_log_pmf_with_covariate_entry_value_matches_advance_prob_entry() -> None:
    """`log_pmf` with a covariate-adjusted `z_entry` uses the same `a_i` as `advance_prob_entry`."""
    family = GeometricDuration(cov_idx=(0,))
    theta = {**_theta([0.05]), "duration.c": jnp.array([[0.3]])}
    d_max = 100
    out = np.asarray(family.log_pmf(theta, d_max=d_max, nominal_dt=1.0 / 252.0, z_entry=jnp.array([0.5])))
    assert np.exp(out[0]).sum() == pytest.approx(1.0, abs=1e-6)

    nb = NegBinDuration(r=(2,), cov_idx=(0,))
    theta_nb = {**_theta([0.05]), "duration.c": jnp.array([[0.3]])}
    out_nb = np.asarray(nb.log_pmf(theta_nb, d_max=d_max, nominal_dt=1.0 / 252.0, z_entry=jnp.array([0.5])))
    assert np.exp(out_nb[0]).sum() == pytest.approx(1.0, abs=1e-6)


def test_e3_t03_negbin_advance_prob_with_covariates() -> None:
    """`NegBinDuration.advance_prob` applies the covariate link when `z_d`/`c` are both given."""
    nb = NegBinDuration(r=(2,), cov_idx=(0,))
    theta = {**_theta([0.05]), "duration.c": jnp.array([[0.3]])}
    dt = jnp.full((3,), 1.0 / 252.0)
    z = jnp.array([[0.0], [1.0], [2.0]])
    out = np.asarray(nb.advance_prob(theta, dt, z))
    assert out[0, 0] < out[1, 0] < out[2, 0]


def test_e3_t03_geometric_mean_sojourn() -> None:
    """`GeometricDuration.mean_sojourn` matches `1 / a` in clock units."""
    theta = _theta([0.05])
    mean_sojourn = float(GEO.mean_sojourn(theta, 1.0 / 252.0)[0])
    expected = (1.0 / 252.0) / (1.0 - np.exp(-0.05 / 252.0))
    assert mean_sojourn == pytest.approx(expected, rel=1e-9)


# --- n_phases --------------------------------------------------------------------------------------


def test_e3_t03_n_phases() -> None:
    """`n_phases` returns 1 for geometric and `r_i` for negbin."""
    assert GEO.n_phases(0) == 1
    nb = NegBinDuration(r=(2, 5))
    assert nb.n_phases(0) == 2
    assert nb.n_phases(1) == 5
