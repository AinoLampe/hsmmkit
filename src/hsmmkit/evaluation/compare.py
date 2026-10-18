"""Comparison of backtests across configurations on one data snapshot (E7-T04).

Concept refs: §3.1 FR-E04, §5.8 results. `comparison_table` has one row per configuration: the summary
metrics per horizon (mean over folds, with the across-fold sd in `<column>_sd`), the in-sample criteria
(`bic`, `icl`: horizon 0), the number of free parameters and the total fit time. `check_snapshots` refuses
to compare runs on different data unless mixing is allowed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from regimekit.evaluation.results import BacktestResult
from regimekit.utils.errors import ConfigError, DataError

LOWER_IS_BETTER = frozenset({"crps", "rmse", "mae", "aic", "bic", "icl", "max_drawdown", "fit_time"})
HIGHER_IS_BETTER = frozenset({"log_score", "log_score_mixture", "sharpe", "hit_rate"})
# components that describe a run instead of ranking it
_ALWAYS = ("aic", "bic", "icl")  # in-sample criteria stay in the table whatever --metric selects
_SKIPPED = ("pit_ks", "pit_coverage.pit", "regime_occupancy", "regime_persistence")


@dataclass(frozen=True)
class CompareEntry:
    """One configuration's backtest: a name, the result and its metadata (`snapshot_ids`, `n_free`, ...)."""

    name: str
    result: BacktestResult
    meta: dict[str, Any]


def check_snapshots(snapshot_ids: Mapping[str, Sequence[str]], allow_mixed: bool) -> None:
    """Refuse to compare runs whose data snapshots differ, unless `allow_mixed`.

    Args:
        snapshot_ids: `{config name: snapshot ids (primary first, then the covariates)}`.
        allow_mixed: accept differing snapshots.

    Raises:
        DataError: the snapshot ids differ (`path="data.cache.snapshot"`).
    """
    distinct = {tuple(ids) for ids in snapshot_ids.values()}
    if len(distinct) > 1 and not allow_mixed:
        listing = "; ".join(f"{name}: {', '.join(i[:12] for i in ids)}" for name, ids in snapshot_ids.items())
        raise DataError(
            f"the configurations use different data snapshots ({listing}); "
            "pass --allow-mixed to compare anyway",
            path="data.cache.snapshot",
        )


def _column(metric: str, horizon: int) -> str:
    return metric if horizon == 0 else f"{metric}@h{horizon}"


def base_metric(column: str) -> str:
    """The metric name of a table column (`log_score@h5` -> `log_score`)."""
    return column.split("@", 1)[0]


def comparison_table(entries: Sequence[CompareEntry], metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per configuration; columns `config, n_free, fit_time`, then `<metric>[@h<h>]` and `_sd`.

    Args:
        entries: the runs to compare (at least one).
        metrics: keep only these metrics (a name matches the metric and its `.component` rows); default all
            that rank a run (the PIT histogram value, the KS rows and the regime shares are left out).

    Raises:
        ConfigError: no entries, or a requested metric is in none of the summaries.
    """
    if not entries:
        raise ConfigError("nothing to compare", path="compare")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        row: dict[str, object] = {
            "config": entry.name,
            "n_free": entry.meta.get("n_free", np.nan),
            "fit_time": float(entry.result.folds["fit_time"].sum()),
        }
        summary = entry.result.summary
        for horizon, metric_name, mean, sd in zip(
            summary["horizon"].tolist(),
            summary["metric"].tolist(),
            summary["mean"].tolist(),
            summary["sd"].tolist(),
            strict=True,
        ):
            name = str(metric_name)
            if name.startswith(_SKIPPED):
                continue
            if metrics is not None and name.split(".", 1)[0] not in (*metrics, *_ALWAYS):
                continue
            col = _column(name, int(horizon))
            seen.add(name.split(".", 1)[0])
            row[col] = float(mean)
            row[f"{col}_sd"] = float(sd)
        rows.append(row)
    missing = [m for m in (metrics or ()) if m not in seen]
    if missing:
        raise ConfigError(f"metric(s) {missing} are in no backtest summary", path="compare.metric")
    frame = pd.DataFrame(rows)
    fixed = ["config", "n_free", "fit_time"]
    rest = sorted((c for c in frame.columns if c not in fixed), key=lambda c: (c.endswith("_sd"), c))
    value_cols = [c for c in rest if not c.endswith("_sd")]
    ordered = [c for col in value_cols for c in (col, f"{col}_sd")]
    return frame[fixed + ordered]


def best_configs(table: pd.DataFrame) -> dict[str, str]:
    """`{column: config with the best value}` for every column whose direction is known.

    Lower is better for `crps, rmse, mae, aic, bic, icl, max_drawdown, fit_time`, higher for `log_score,
    log_score_mixture, sharpe, hit_rate`; other columns (coverage, turnover, `n_free`) are not ranked. With a
    single configuration nothing is marked.
    """
    best: dict[str, str] = {}
    if len(table) < 2:
        return best
    for col in table.columns:
        if col in ("config", "n_free") or str(col).endswith("_sd"):
            continue
        name = base_metric(str(col))
        values = table[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).any() or not (name in LOWER_IS_BETTER or name in HIGHER_IS_BETTER):
            continue
        pick = np.nanargmin(values) if name in LOWER_IS_BETTER else np.nanargmax(values)
        best[str(col)] = str(table["config"].iloc[int(pick)])
    return best


def _fmt(mean: float, sd: float | None) -> str:
    if not np.isfinite(mean):
        return "n/a"
    text = f"{mean:.4g}"
    return text if sd is None or not np.isfinite(sd) else f"{text} ± {sd:.2g}"


def render_markdown(table: pd.DataFrame, snapshot_ids: Mapping[str, Sequence[str]]) -> str:
    """`compare.md`: the snapshot ids and the table as `mean ± sd`, the best value of each column in bold."""
    best = best_configs(table)
    value_cols = [c for c in table.columns if c not in ("config",) and not str(c).endswith("_sd")]
    lines = [
        "# Backtest comparison",
        "",
        "## Data snapshots",
        "",
        *[f"- `{name}`: {', '.join(f'`{i[:12]}`' for i in ids)}" for name, ids in snapshot_ids.items()],
        "",
        "## Results",
        "",
        "Mean over folds ± sd across folds; the best value of each ranked column is **bold**. "
        "`bic`/`icl` are in-sample (per fold, mean over folds); `@h<k>` columns are out-of-sample "
        "at horizon `k`.",
        "",
        "| config | " + " | ".join(str(c) for c in value_cols) + " |",
        "|---|" + "---|" * len(value_cols),
    ]
    for _, row in table.iterrows():
        cells = []
        for col in value_cols:
            sd = row[f"{col}_sd"] if f"{col}_sd" in table.columns else None
            text = _fmt(float(row[col]), None if sd is None else float(sd))
            cells.append(f"**{text}**" if best.get(str(col)) == row["config"] else text)
        lines.append(f"| {row['config']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
