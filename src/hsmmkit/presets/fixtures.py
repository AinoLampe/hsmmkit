"""Synthetic data generators of the core presets (E10-T10): `fixture(seed) -> {relative path: DataFrame}`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "R7_TRUTH",
    "R8_TRUTH",
    "R9_TRUTH",
    "R10_TRUTH",
    "R11_TRUTH",
    "segmentation_gaussian_ltr",
    "simulate_r7",
    "simulate_r8",
    "simulate_r9",
    "simulate_r10",
    "simulate_r11",
    "two_regime_gaussian",
]

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


# --- recovery scenarios R7-R11 (E14-T06): deterministic NumPy generators, truth in the `R*_TRUTH` dicts -----

R7_TRUTH: dict[str, Any] = {
    "mean": np.array([[-2.0, -1.0, 0.0], [0.0, 2.0, -1.0], [2.0, 0.0, 1.5]]),
    "chol": np.array(
        [
            [[0.7, 0.0, 0.0], [0.3, 0.6, 0.0], [-0.2, 0.2, 0.7]],
            [[0.8, 0.0, 0.0], [-0.4, 0.7, 0.0], [0.2, 0.3, 0.6]],
            [[0.6, 0.0, 0.0], [0.2, 0.8, 0.0], [0.3, -0.3, 0.7]],
        ]
    ),
    "sojourn": np.array([12.0, 16.0, 20.0]),
    "r": 2,
    "pi_rows": np.array([[0.7, 0.3], [0.4, 0.6], [0.3, 0.7]]),
}
R8_TRUTH: dict[str, Any] = {
    "rate_per_step": np.array([3.0, 12.0]),
    "shape": np.array([3.0, 4.0]),
    "scale_steps": np.array([4.0, 6.0]),
    "dt_days": np.array([1.0, 1.0, 1.0, 1.0, 3.0]),
}
R9_TRUTH: dict[str, Any] = {
    "mean": np.array([-0.5, 1.0]),
    "phi": np.array([0.35, -0.2]),
    "sigma": 0.8,
    "sojourn": np.array([10.0, 20.0]),
}
R10_TRUTH: dict[str, Any] = {
    "mean": np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-2.0, -2.0]]),
    "sd": 0.8,
    "shape": np.array([3.0, 3.0]),
    "scale": np.array([3.0, 5.0]),
    "geometric_mean": 12.0,
    "label_fraction": 0.3,
}
R11_TRUTH: dict[str, Any] = {"mean": np.array([-2.0, 0.0, 2.0]), "sd": 0.5, "sojourn": 40.0}


def _jump_path(
    rng: np.random.Generator,
    length: int,
    jump: np.ndarray,
    draw: Callable[[int, np.random.Generator], int],
    start: int | None = None,
) -> np.ndarray:
    """A regime path of `length` steps: `draw(state, rng) -> duration >= 1`, jumps from the rows of `jump`."""
    n = jump.shape[0]
    out = np.empty(length, dtype=np.int64)
    state = int(rng.integers(n)) if start is None else start
    t = 0
    while t < length:
        d = int(draw(state, rng))
        out[t : t + d] = state
        t += d
        state = int(rng.choice(n, p=jump[state]))
    return out


def _full_jump(pi_rows: np.ndarray) -> np.ndarray:
    """`(N, N-1)` target probabilities (self excluded, ascending) -> the `(N, N)` jump matrix."""
    n = pi_rows.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        out[i, np.arange(n) != i] = pi_rows[i]
    return out


def simulate_r7(seed: int, n_seq: int = 50, t_len: int = 300) -> pd.DataFrame:
    """R7: `n_seq` sequences of a 3-regime, 3-series full-covariance Gaussian with NB (`r = 2`) durations."""
    rng = np.random.default_rng(seed)
    tr = R7_TRUTH
    jump = _full_jump(tr["pi_rows"])
    p_phase = tr["r"] / tr["sojourn"]

    def draw(state: int, g: np.random.Generator) -> int:
        return int(g.geometric(p_phase[state], size=tr["r"]).sum())

    parts = []
    for b in range(n_seq):
        z = _jump_path(rng, t_len, jump, draw)
        noise = rng.normal(size=(t_len, 3))
        y = tr["mean"][z] + np.einsum("tij,tj->ti", tr["chol"][z], noise)
        parts.append(
            pd.DataFrame(
                {"sequence_id": b, "t": np.arange(t_len), "y0": y[:, 0], "y1": y[:, 1], "y2": y[:, 2]}
            )
        )
    return pd.concat(parts, ignore_index=True)


def simulate_r8(seed: int, t_len: int = 8000) -> pd.DataFrame:
    """R8: Poisson counts, 2 regimes with gamma durations, irregular calendar grid (`dt` in years)."""
    rng = np.random.default_rng(seed)
    tr = R8_TRUTH
    jump = np.array([[0.0, 1.0], [1.0, 0.0]])

    def draw(state: int, g: np.random.Generator) -> int:
        return max(1, int(np.ceil(g.gamma(tr["shape"][state], tr["scale_steps"][state]))))

    z = _jump_path(rng, t_len, jump, draw)
    gap_days = rng.choice(tr["dt_days"], size=t_len)
    dt = gap_days / 365.25
    y = rng.poisson(tr["rate_per_step"][z] * gap_days).astype(np.float64)
    return pd.DataFrame({"t": np.arange(t_len), "dt": dt, "y": y, "regime": z})


def simulate_r9(seed: int, t_len: int = 5000) -> pd.DataFrame:
    """R9: a 2-regime Hamilton-type mean-adjusted MS-AR(2) series."""
    rng = np.random.default_rng(seed)
    tr = R9_TRUTH
    state, y, path = 0, np.zeros(t_len), np.zeros(t_len, dtype=np.int64)
    for t in range(t_len):
        path[t] = state
        mean = tr["mean"][state]
        for lag in (1, 2):
            if t - lag >= 0:
                mean += tr["phi"][lag - 1] * (y[t - lag] - tr["mean"][path[t - lag]])
        y[t] = mean + tr["sigma"] * rng.normal()
        if rng.random() < 1.0 / tr["sojourn"][state]:
            state = 1 - state
    return pd.DataFrame({"t": np.arange(t_len), "y": y, "regime": path})


def simulate_r10(seed: int, n_seq: int = 60) -> pd.DataFrame:
    """R10: `n_seq` left-to-right sessions 0, 1, 2, 3 (regime 3 absorbing), 30 % of the steps labelled.

    Durations: gamma (regime 0), gamma (1), geometric (2); a session of random horizon (80-160 steps) is cut
    wherever it stands, so the last segment is right-censored independently of its length.
    """
    rng = np.random.default_rng(seed)
    tr = R10_TRUTH
    parts = []
    for b in range(n_seq):
        horizon = int(rng.integers(80, 161))
        regimes: list[int] = []
        for k in range(3):
            if k == 2:
                d = int(rng.geometric(1.0 / tr["geometric_mean"]))
            else:
                d = max(1, int(np.ceil(rng.gamma(tr["shape"][k], tr["scale"][k]))))
            regimes.extend([k] * d)
        regimes.extend([3] * horizon)
        z = np.asarray(regimes[:horizon])
        y = tr["mean"][z] + tr["sd"] * rng.normal(size=(horizon, 2))
        labelled = rng.random(horizon) < tr["label_fraction"]
        parts.append(
            pd.DataFrame(
                {
                    "sequence_id": b,
                    "t": np.arange(horizon),
                    "y0": y[:, 0],
                    "y1": y[:, 1],
                    "label": np.where(labelled, z.astype(np.float64), np.nan),
                    "regime": z,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def simulate_r11(seed: int, t_len: int = 5000) -> pd.DataFrame:
    """R11: three well-separated Gaussian regimes (the data of a 4-regime VB fit that must prune one)."""
    rng = np.random.default_rng(seed)
    tr = R11_TRUTH
    z = np.empty(t_len, dtype=np.int64)
    state = 0
    for t in range(t_len):
        z[t] = state
        if rng.random() < 1.0 / tr["sojourn"]:
            state = int(rng.choice([k for k in range(3) if k != state]))
    y = rng.normal(tr["mean"][z], tr["sd"])
    return pd.DataFrame({"t": np.arange(t_len), "y": y, "regime": z})
