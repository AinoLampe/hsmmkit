"""E7-T02b: `Preparer.fit(train_start=)` restricts the standardization window from below."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regimekit.data.prepare import Preparer
from tests.helpers import backtest_fixtures as bf


def test_e7_t02b_train_start_limits_the_statistics_and_the_recorded_window(tmp_path: Path) -> None:
    """Moments come from rows in `[train_start, train_end]` only; default behaviour is unchanged."""
    returns = bf.simulate_returns(120, seed=2)
    obs = bf.observations(returns)
    cov = bf.covariate(obs, np.concatenate([[0.0], returns]))
    cfg = bf.make_config(tmp_path, with_covariate=True)
    raw_cfg = cfg.model_copy(deep=True)
    raw_cfg.data.preparation.standardize_covariates = False
    raw = np.asarray(
        Preparer(raw_cfg.data).fit(obs, [cov], obs.times[-1]).transform(obs, [cov]).cov_emission
    )[:, 0]
    times = Preparer(cfg.data).fit(obs, [cov], obs.times[-1]).transform(obs, [cov]).times

    whole = Preparer(cfg.data).fit(obs, [cov], train_end=times[79])
    assert pd.Timestamp(whole.prep_stats.train_window[0]) == pd.Timestamp(times[0])
    np.testing.assert_allclose(whole.prep_stats.cov_mean, [raw[:80].mean()], rtol=1e-12, atol=1e-15)

    window = Preparer(cfg.data).fit(obs, [cov], train_end=times[79], train_start=times[30])
    assert pd.Timestamp(window.prep_stats.train_window[0]) == pd.Timestamp(times[30])
    np.testing.assert_allclose(window.prep_stats.cov_mean, [raw[30:80].mean()], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(window.prep_stats.cov_std, [raw[30:80].std()], rtol=1e-12)
