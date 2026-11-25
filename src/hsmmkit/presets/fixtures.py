"""Synthetic data generators of the core presets (E10-T10): `fixture(seed) -> {relative path: DataFrame}`."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["segmentation_gaussian_ltr", "two_regime_gaussian"]

_LTR_MEANS = np.array([[0.0, 0.0, 1.0], [1.5, 0.0, 0.0], [0.0, 1.5, -1.0], [-1.5, -1.5, 0.5]])
_LTR_MEAN_SOJOURN = (35.0, 50.0, 15.0, 40.0)


def segmentation_gaussian_ltr(seed: int = 0) -> dict[str, pd.DataFrame]:
    """50 labelled sessions of 3-dim Gaussian windows with a left-to-right 4-regime structure.

    Session lengths are drawn from 150-400 steps; regimes are visited in the order 0, 1, 2, 3 (a session ends
    in whatever regime it has reached); about 40 % of the labels are given, the rest are NaN (unlabelled).
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for session in range(50):
        length = int(rng.integers(150, 401))
        regime = np.empty(length, dtype=np.int64)
        t, k = 0, 0
        while t < length:
            stay = 1 + int(rng.geometric(1.0 / _LTR_MEAN_SOJOURN[k]))
            regime[t : t + stay] = k
            t += stay
            k = min(k + 1, 3)
        obs = _LTR_MEANS[regime] + rng.normal(0.0, 0.6, size=(length, 3))
        labelled = rng.random(length) < 0.4
        parts.append(
            pd.DataFrame(
                {
                    "session_id": session,
                    "acc_x": obs[:, 0],
                    "acc_y": obs[:, 1],
                    "acc_z": obs[:, 2],
                    "activity": np.where(labelled, regime.astype(np.float64), np.nan),
                }
            )
        )
    return {"data/sensor_windows.parquet": pd.concat(parts, ignore_index=True)}


def two_regime_gaussian(seed: int = 0) -> dict[str, pd.DataFrame]:
    """A 1500-step daily series from a calm and a turbulent Gaussian regime (per-step parameters)."""
    rng = np.random.default_rng(seed)
    n = 1500
    mean, scale, leave = (0.0003, -0.00064), (0.00756, 0.02205), (0.0079, 0.0235)
    state, x = 0, np.empty(n)
    for t in range(n):
        x[t] = rng.normal(mean[state], scale[state])
        if rng.random() < leave[state]:
            state = 1 - state
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    return {"data/two_regime.csv": pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "x": x})}
