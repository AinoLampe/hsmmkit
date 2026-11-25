"""E10-T07a: walk-forward origin hooks and a core backtest without hooks (no finance import)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from regimekit.data.contracts import PreparedDataset
from regimekit.data.prepare import Preparer
from regimekit.engine.base import Posterior
from regimekit.evaluation.backtest import OriginHook, run_backtest, schedule_folds
from regimekit.evaluation.metrics.registry import FunctionMetric, register
from regimekit.inference.results import ForecastResult
from tests.helpers import core_config as cc


@dataclass
class RecordingHook:
    """Records every origin and gives a one-column frame (an `OriginHook`)."""

    name: str = "recorder"
    calls: list[tuple[int, int, int, int]] = field(default_factory=list)

    def on_origin(
        self,
        fold: int,
        origin: int,
        posterior: Posterior,
        forecast: ForecastResult | None,
        data: PreparedDataset,
    ) -> dict[str, Any]:
        """Remember `(fold, origin, filter rows, T)` and give a value that depends on the origin."""
        assert forecast is not None
        self.calls.append((fold, origin, int(posterior.log_alpha.shape[0]), int(data.observations.shape[0])))
        return {} if origin % 7 == 0 else {"value": float(origin)}

    def finalize(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        """The rows as a frame."""
        return pd.DataFrame(rows)


SEEN: list[pd.DataFrame] = []


def recorder_total(**ctx: Any) -> float:  # noqa: ANN401
    """A metric on the recorder hook's frame (one fold's slice)."""
    frame = ctx["extras"]["recorder"]
    SEEN.append(frame)
    return float(frame["value"].sum())


register(FunctionMetric("recorder_total", frozenset({"recorder"}), recorder_total), replace=True)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A core config, its observations and the fold table."""
    tmp = tmp_path_factory.mktemp("core")
    cfg = cc.core_config(tmp, metrics=("log_score", "bic", "recorder_total"))
    obs = cc.observations(cc.simulate_values())
    grid = Preparer(cfg.data).fit(obs, [], train_end=obs.require_times()[-1]).transform(obs, [])
    folds = schedule_folds(np.asarray(grid.times), np.asarray(grid.dt), cfg.evaluation.backtest, 1)
    return {"cfg": cfg, "obs": obs, "folds": folds}


def test_e10_t07a_core_backtest_without_hooks_runs_on_a_gaussian_config(world: dict[str, Any]) -> None:
    """AC2: no hooks, a `ProjectConfig` with `gaussian` and no finance import: metrics and empty extras."""
    result = run_backtest(world["cfg"], lambda: (world["obs"], []), seed=5)
    assert result.extras == {}
    assert set(result.metrics["metric"]) == {"log_score", "bic"}  # `recorder_total` needs a hook: skipped
    assert result.folds.shape[0] == len(world["folds"])
    assert np.isfinite(result.metrics["value"]).all()


def test_e10_t07a_a_hook_sees_every_origin_and_its_frame_lands_in_extras(world: dict[str, Any]) -> None:
    """The hook is called at each origin with the filter rows up to it; `finalize` fills `extras[name]`."""
    hook = RecordingHook()
    assert isinstance(hook, OriginHook)
    SEEN.clear()
    result = run_backtest(world["cfg"], lambda: (world["obs"], []), seed=5, hooks=[hook])
    origins = [(f.fold, o) for f in world["folds"] for o in f.origins()]
    assert [(c[0], c[1]) for c in hook.calls] == origins
    assert all(rows == origin + 1 for _, origin, rows, _ in hook.calls)  # filter rows up to the origin
    frame = result.extras["recorder"]
    assert list(frame.columns) == ["fold", "origin", "value"]
    assert len(frame) == len([o for _, o in origins if o % 7 != 0])  # `{}` means "no row at this origin"
    per_fold = result.metrics[result.metrics["metric"] == "recorder_total"]
    assert (per_fold["horizon"] == 0).all()
    for fold in world["folds"]:
        expected = float(frame.loc[frame["fold"] == fold.fold, "value"].sum())
        got = float(per_fold.loc[per_fold["fold"] == fold.fold, "value"].iloc[0])
        assert got == expected
    assert all(set(f["fold"]) == {int(f["fold"].iloc[0])} for f in SEEN)  # one fold's slice per call


def test_e10_t07a_results_with_and_without_a_hook_have_the_same_model_metrics(world: dict[str, Any]) -> None:
    """A hook observes; it does not change the fits or the forecasts."""
    plain = run_backtest(world["cfg"], lambda: (world["obs"], []), seed=5)
    hooked = run_backtest(world["cfg"], lambda: (world["obs"], []), seed=5, hooks=[RecordingHook()])
    keep = plain.metrics["metric"].unique()
    got = hooked.metrics[hooked.metrics["metric"].isin(keep)].reset_index(drop=True)
    pd.testing.assert_frame_equal(plain.metrics, got, check_exact=True)
