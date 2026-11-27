"""E15-T02 AC3: the per-call cost of `smoothed_at` is linear in the lag (`-m benchmark`)."""

from __future__ import annotations

import time

import jax
import numpy as np
import pytest

from hsmmkit.online import ExpandedOnlineFilter
from tests.helpers.engine_fixtures import fit_of, make_scenario

pytestmark = pytest.mark.benchmark

LAGS = (10, 20, 40, 80)


def _best_call_seconds(lag: int) -> float:
    sc = make_scenario(3, [4, 4, 4], 1, 400, n_cov_pi=2, n_cov_d=2, seed=0)  # S = 12
    filt = ExpandedOnlineFilter(fit_of(sc), lag=lag)  # type: ignore[arg-type]
    d = sc.data
    z = {"transition": np.asarray(d.cov_transition), "duration": np.asarray(d.cov_duration)}
    state, *_ = filt.update_many(filt.start(), np.asarray(d.observations), np.asarray(d.dt), z)
    jax.block_until_ready(filt.smoothed_at(state))  # compile
    samples = []
    for _ in range(1500):
        t0 = time.perf_counter()
        jax.block_until_ready(filt.smoothed_at(state))
        samples.append(time.perf_counter() - t0)
    return float(np.min(samples))  # the minimum is the low-noise estimator of a fixed-cost kernel


def test_e15_t02_smoothed_at_cost_is_linear_in_the_lag() -> None:
    """Best-of-1500 per-call time at L in {10, 20, 40, 80} fits a line with R^2 >= 0.95."""
    times = np.array([_best_call_seconds(lag) for lag in LAGS])
    slope, intercept = np.polyfit(LAGS, times, 1)
    pred = slope * np.asarray(LAGS) + intercept
    r2 = 1.0 - float(np.sum((times - pred) ** 2) / np.sum((times - times.mean()) ** 2))
    print(
        "E15-T02 smoothed_at best [us]:",
        dict(zip(LAGS, np.round(times * 1e6, 1), strict=True)),
        f"R2={r2:.4f}",
    )
    assert slope > 0
    assert r2 >= 0.95
