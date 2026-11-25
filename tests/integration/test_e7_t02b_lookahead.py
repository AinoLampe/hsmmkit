"""E7-T02b AC1: the official walk-forward pipeline does not leak a covariate's future into forecasts.

Setup. A covariate that is a *perfect* contemporaneous signal: the value stamped at `tau_s` is the return
`y_s` realized at `tau_s` (information that exists at `tau_s`, so it can only be used for steps after `s`).
With `lag: 1` the covariate row of step `s` is the last print at or before `tau_{s-1}` (CLAUDE.md 3.7), i.e.
`y_{s-1}`: past information, no use for predicting `y_s` of a model whose returns are conditionally
independent given the regime. The out-of-sample log score with the covariate must therefore not beat the
covariate-free model.

Sensitivity. The same series stamped one step *earlier* (`y_s` at `tau_{s-1}`) is a lag-0 leak that the
official pipeline cannot know about (it is a mis-stamped input, equivalent to injecting `lag=0` into the
arrays). The fit exploits it (row `s` explains `y_s` exactly: the volatility collapses), and the forecast
-- which for the first step only has the last *observed* covariate row (E6-T04b), not the leaked row --
then scores catastrophically. That is how a leak in the training data shows up out-of-sample in this
pipeline, and it separates the two variants by many orders of magnitude, so the comparison is sensitive.
(The ticket expected a large *gain* here; the stand-in covariate row of the first forecast step makes a
gain impossible, see STATUS Findings.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from regimekit.data.contracts import CovariateSeries, ObservationSet
from regimekit.evaluation.backtest import run_backtest
from tests.helpers import backtest_fixtures as bf

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

CHEAP = {"metrics": ("log_score",), "n_sim": 100, "horizons": (1,), "restarts": 1}


def _log_score_by_fold(
    tmp_path: Path, obs: ObservationSet, covs: list[CovariateSeries], *, with_covariate: bool
) -> np.ndarray:
    cfg = bf.make_config(tmp_path, with_covariate=with_covariate, **CHEAP)  # type: ignore[arg-type]
    metrics = run_backtest(cfg, lambda: (obs, covs), seed=5).metrics
    rows = metrics[(metrics["metric"] == "log_score") & (metrics["horizon"] == 1)].sort_values("fold")
    return rows["value"].to_numpy()


def test_e7_t02b_lookahead_covariate_with_lag_one_carries_no_out_of_sample_gain(tmp_path: Path) -> None:
    """AC1: a properly stamped perfect signal gives no gain (paired t-test, 5 %); a leak is detected."""
    returns = bf.simulate_returns(bf.N_PRICES - 1)
    obs = bf.observations(returns)
    proper = bf.covariate(obs, np.concatenate([[0.0], returns]))  # y_s stamped at tau_s
    leaked = bf.covariate(obs, returns)  # y_s stamped at tau_{s-1}: lag 0 in effect

    base = _log_score_by_fold(tmp_path, obs, [], with_covariate=False)
    with_signal = _log_score_by_fold(tmp_path, obs, [proper], with_covariate=True)
    with_leak = _log_score_by_fold(tmp_path, obs, [leaked], with_covariate=True)
    assert base.shape == with_signal.shape == with_leak.shape == (4,)

    # no gain: H1 "the covariate improves the score" is not accepted at 5 %
    gain = stats.ttest_1samp(with_signal - base, 0.0, alternative="greater")
    assert gain.pvalue > 0.05, f"covariate improved the OOS log score: diffs {with_signal - base}"
    # the score is also not meaningfully different (the model should ignore a useless covariate)
    assert np.abs(with_signal - base).max() < 0.1

    # sensitivity: with the leak in the arrays every fold collapses by far more than any noise
    assert np.all(with_leak - base < -1.0), f"leak not detected: diffs {with_leak - base}"


def test_e7_t02b_lookahead_helper_stamps_are_what_the_docstring_says() -> None:
    """The two covariates differ exactly by one grid step of stamping (guards the test's own setup)."""
    returns = bf.simulate_returns(30)
    obs = bf.observations(returns)
    proper = bf.covariate(obs, np.concatenate([[0.0], returns]))
    leaked = bf.covariate(obs, returns)
    # value y_s: stamped tau_s in `proper`, tau_{s-1} in `leaked`
    for s in (1, 7, 30):
        assert proper.values[s] == returns[s - 1] == leaked.values[s - 1]
        assert proper.times[s] == obs.times[s]
        assert leaked.times[s - 1] == obs.times[s - 1]
