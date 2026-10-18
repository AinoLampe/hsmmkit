"""`report.md` of a backtest run (E7-T04).

Concept refs: §5.9, §4.10. The report is deterministic for a given run (no timestamps): header (config,
snapshots, seed, versions, backend), model summary, per-fold fit summary, diagnostics of the last fold's fit
with their warnings (E5-T04), the backtest summary table, the strategy table and timings.
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import pandas as pd

from regimekit.evaluation.metrics.strategy import hit_rate, max_drawdown, sharpe, turnover

if TYPE_CHECKING:
    from regimekit.config.schema import ProjectConfig
    from regimekit.evaluation.results import BacktestResult

SECTIONS = (
    "Header",
    "Model",
    "Fit",
    "Diagnostics",
    "Backtest summary",
    "Strategy",
    "Timings",
)


def _md_table(frame: pd.DataFrame) -> str:
    """A pipe table of `frame` (floats to 4 significant digits)."""

    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "n/a" if not np.isfinite(value) else f"{value:.4g}"
        return str(value)

    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def versions() -> dict[str, str]:
    """Versions and backend of the running process (recorded in `backtest.json` and the header)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        package = version("regimekit")
    except PackageNotFoundError:  # running from a source tree without an installed distribution
        package = "unknown"
    return {
        "regimekit": package,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "x64": str(bool(jax.config.jax_enable_x64)),
    }


def _model_lines(cfg: ProjectConfig, meta: dict[str, Any]) -> list[str]:
    m = cfg.model
    covs = [f"`{c.name}` ({', '.join(c.use_in)})" for c in cfg.data.covariates] or ["none"]
    lines = [
        f"- regimes N = {m.n_states}, engine `{m.engine}`, transition order {m.transition.order}, "
        f"mode `{m.transition.mode}`, conditioning `{m.transition.duration_conditioning}`",
        f"- emission `{m.emission.family}`, duration `{m.duration.family}`"
        + (f" (r = {m.duration.r})" if m.duration.r else ""),
        f"- covariates: {', '.join(covs)}",
        f"- clock `{cfg.data.clock}`, free parameters: {meta.get('n_free', 'n/a')}",
    ]
    return lines


def _diagnostic_lines(diagnostics: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """`(lines, warnings)` from the `{"value", "warning"}` entries of `estimation.report.diagnose`."""
    if diagnostics is None:
        return ["No diagnostics were computed."], []
    lines: list[str] = []
    warns: list[str] = []
    for name, entry in diagnostics.items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        warning = entry.get("warning") if isinstance(entry, dict) else None
        arr = np.asarray(value) if value is not None and not isinstance(value, dict | str) else None
        shown: object = value
        if arr is not None and arr.ndim > 0 and arr.size > 6:
            shown = f"array{tuple(arr.shape)}"
        elif arr is not None and arr.ndim > 0:
            shown = np.array2string(arr, precision=4)
        lines.append(f"- `{name}`: {shown}" + (f" **warning:** {warning}" if warning else ""))
        if warning:
            warns.append(f"{name}: {warning}")
    return lines, warns


def _strategy_lines(result: BacktestResult, cfg: ProjectConfig) -> list[str]:
    trace = result.strategy
    if trace is None:
        return ["No strategy was configured."]
    s = cfg.evaluation.strategy
    overall = {
        "total log-return": float(trace["return"].sum()),
        "final equity": float(trace["equity"].iloc[-1]),
        "sharpe": sharpe(strategy=trace),
        "max_drawdown": max_drawdown(strategy=trace),
        "turnover": turnover(strategy=trace),
        "hit_rate": hit_rate(strategy=trace),
    }
    desc = f"`{s.type}`, cost {s.cost_bps:g} bp" if s is not None else "configured strategy"
    table = pd.DataFrame({"statistic": list(overall), "value": list(overall.values())})
    return [
        f"Strategy {desc}; {len(trace)} out-of-sample periods in {trace['fold'].nunique()} folds.",
        "",
        _md_table(table),
    ]


def build_report(
    name: str,
    cfg: ProjectConfig,
    result: BacktestResult,
    meta: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> str:
    """The Markdown report of one backtest run (every heading of `SECTIONS`)."""
    snaps = meta.get("snapshot_ids", [])
    ver = meta.get("versions", {})
    folds = result.folds
    fit_cols = ["fold", "train_end", "objective", "loglik", "converged", "n_iter", "restart_gap", "fit_time"]
    fit_table = folds[[c for c in fit_cols if c in folds.columns]]
    converged = int(folds["converged"].sum()) if "converged" in folds.columns else 0
    diag_lines, warns = _diagnostic_lines(diagnostics)
    summary = result.summary.rename(columns={"mean": "mean (over folds)", "sd": "sd (across folds)"})
    fit_total = float(folds["fit_time"].sum())
    best_fold = int(folds["fold"].iloc[int(np.argmin(folds["objective"].to_numpy()))])

    out = [
        f"# Backtest report: {name}",
        "",
        "## Header",
        "",
        f"- config: `{name}`, seed {meta.get('seed', 'n/a')}",
        f"- snapshot ids: {', '.join(f'`{s}`' for s in snaps) if snaps else 'n/a'}",
        f"- versions: regimekit {ver.get('regimekit', 'n/a')}, python {ver.get('python', 'n/a')}, "
        f"jax {ver.get('jax', 'n/a')}",
        f"- backend: {ver.get('backend', 'n/a')}, x64 {ver.get('x64', 'n/a')}",
        f"- scheme `{cfg.evaluation.backtest.scheme}`, {len(folds)} folds, "
        f"horizons {cfg.evaluation.backtest.horizons}",
        "",
        "## Model",
        "",
        *_model_lines(cfg, meta),
        "",
        "## Fit",
        "",
        f"{converged} of {len(folds)} fold fits converged; best objective "
        f"{float(folds['objective'].min()):.6g} (fold {best_fold}); "
        "largest gap between the two best restarts "
        f"{float(folds['restart_gap'].max()) if 'restart_gap' in folds.columns else float('nan'):.4g} "
        "(0 with a single restart).",
        "",
        _md_table(fit_table),
        "",
        "## Diagnostics",
        "",
        "Fit diagnostics (E5-T04) of the last fold's fit on its training window.",
        "",
        *diag_lines,
        "",
        *(["**Warnings**", "", *[f"- {w}" for w in warns]] if warns else ["No diagnostic warnings."]),
        "",
        "## Backtest summary",
        "",
        "Rows with horizon 0 are in-sample (per fold) or strategy metrics; `pit_ks` rows pool all origins.",
        "",
        _md_table(summary),
        "",
        "## Strategy",
        "",
        *_strategy_lines(result, cfg),
        "",
        "## Timings",
        "",
        f"- total fit time {fit_total:.1f} s, mean per fold {fit_total / max(len(folds), 1):.1f} s",
        f"- total backtest wall time {meta.get('wall_s', float('nan')):.1f} s",
    ]
    return "\n".join(out) + "\n"
