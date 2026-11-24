"""Data-quality report written by `regimekit fetch` (FR-D06, concept §5.6).

Checks, all on the raw primary prices as fetched (before the master grid is built):

- gaps against a plain weekday calendar (daily data only; no holiday calendar in 0.1.0, so exchange
  holidays show up as gaps by design),
- duplicated timestamps,
- zero or negative prices,
- `|return| > 8 sigma` log-returns (flag only; sigma is the robust MAD scale `1.4826 * MAD`, so a single
  outlier cannot hide itself by inflating the standard deviation),
- adjustment discontinuities between the two latest cache generations (max relative price
  difference on their overlap `> 1e-6`),
- per-covariate staleness histogram and the rows dropped by each missing-value policy (from the
  `Preparer`'s `PrepDiagnostics`).

A series whose MAD is 0 (e.g. mostly unchanged prices) gets no outlier check.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regimekit.data.contracts import ObservationSet

OUTLIER_SIGMAS = 8.0
ADJUSTMENT_RTOL = 1e-6
_MAD_TO_SIGMA = 1.4826


def _iso(t: np.datetime64 | pd.Timestamp) -> str:
    ts = pd.Timestamp(t)
    return str(ts.date()) if ts == ts.normalize() else ts.isoformat()


@dataclass(frozen=True)
class DataQualityReport:
    """Findings of the data-quality checks; every finding carries its timestamp(s)."""

    symbols: list[str]
    n_rows: int
    gaps: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    non_positive_prices: list[dict[str, Any]] = field(default_factory=list)
    outliers: list[dict[str, Any]] = field(default_factory=list)
    adjustment: dict[str, Any] | None = None
    staleness: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_rows: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        """The report as indented JSON."""
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        """The report as a human-readable Markdown page."""
        lines = [
            "# Data-quality report",
            "",
            f"Symbols: {', '.join(self.symbols)} · rows: {self.n_rows}",
            "",
            f"## Gaps vs. weekday calendar ({len(self.gaps)})",
            "",
            *(f"- {d}" for d in self.gaps),
            "",
            f"## Duplicated timestamps ({len(self.duplicates)})",
            "",
            *(f"- {d}" for d in self.duplicates),
            "",
            f"## Zero or negative prices ({len(self.non_positive_prices)})",
            "",
            *(f"- {p['time']} {p['symbol']}: {p['price']}" for p in self.non_positive_prices),
            "",
            f"## |return| > {OUTLIER_SIGMAS:g} sigma, flag only ({len(self.outliers)})",
            "",
            *(
                f"- {o['time']} {o['symbol']}: {o['return']:+.6f} ({o['sigmas']:.1f} sigma)"
                for o in self.outliers
            ),
            "",
            "## Adjustment discontinuities between cache generations",
            "",
        ]
        if self.adjustment is None:
            lines.append("- only one cache generation; nothing to compare")
        else:
            a = self.adjustment
            verdict = "DISCONTINUOUS" if a["discontinuous"] else "consistent"
            first = f", first at {a['first_time']}" if a["first_time"] else ""
            lines.append(
                f"- {verdict}: {a['latest'][:12]} vs {a['previous'][:12]}, "
                f"{a['n_overlap']} overlapping rows, "
                f"max relative difference {a['max_rel_diff']:.3g}{first}"
            )
        lines += ["", "## Covariate staleness (grid steps since the print became visible)", ""]
        for name, hist in self.staleness.items():
            lines.append(f"- {name}: " + ", ".join(f"{k}: {v}" for k, v in hist.items()))
        lines += ["", "## Dropped rows", ""]
        lines += [f"- {k}: {v}" for k, v in self.dropped_rows.items()]
        return "\n".join(lines) + "\n"

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        """Write `quality.json` and `quality.md` into `out_dir` (created if missing)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path, md_path = out_dir / "quality.json", out_dir / "quality.md"
        json_path.write_text(self.to_json(), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path


def _gaps(times: np.ndarray, interval: str) -> list[str]:
    if interval != "1d" or times.size == 0:
        return []
    days = pd.DatetimeIndex(times).normalize().unique()
    expected = pd.bdate_range(days.min(), days.max())
    return [_iso(d) for d in expected.difference(days)]


def _duplicates(times: np.ndarray) -> list[str]:
    index = pd.DatetimeIndex(times)
    return [_iso(t) for t in index[index.duplicated(keep="first")].unique()]


def _times(obs: ObservationSet) -> np.ndarray:
    return np.asarray(obs.require_times())


def _non_positive(obs: ObservationSet) -> list[dict[str, Any]]:
    found = []
    times = _times(obs)
    rows, cols = np.nonzero(np.asarray(obs.values) <= 0)
    for r, c in sorted(zip(rows.tolist(), cols.tolist(), strict=True), key=lambda rc: (times[rc[0]], rc[1])):
        found.append(
            {"time": _iso(times[r]), "symbol": obs.series_names[c], "price": float(obs.values[r, c])}
        )
    return found


def _outliers(obs: ObservationSet) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    frame = pd.DataFrame(np.asarray(obs.values, dtype=np.float64), index=pd.DatetimeIndex(_times(obs)))
    frame = frame.sort_index(kind="stable")
    frame = frame[~frame.index.duplicated(keep="last")]
    for c, symbol in enumerate(obs.series_names):
        series = frame.iloc[:, c]
        series = series[np.isfinite(series) & (series > 0)]
        returns = np.log(series).diff().dropna()
        if returns.size < 3:
            continue
        deviations = np.abs(returns - returns.median())
        sigma = _MAD_TO_SIGMA * float(deviations.median())
        if sigma <= 0:
            continue
        for t, r in returns[np.abs(returns) > OUTLIER_SIGMAS * sigma].items():
            found.append(
                {"time": _iso(t), "symbol": symbol, "return": float(r), "sigmas": float(abs(r) / sigma)}
            )
    return sorted(found, key=lambda o: (o["time"], o["symbol"]))


def _adjustment(latest: ObservationSet, previous: ObservationSet) -> dict[str, Any]:
    new = pd.DataFrame(np.asarray(latest.values, dtype=np.float64), index=pd.DatetimeIndex(_times(latest)))
    old = pd.DataFrame(
        np.asarray(previous.values, dtype=np.float64), index=pd.DatetimeIndex(_times(previous))
    )
    new = new[~new.index.duplicated(keep="last")]
    old = old[~old.index.duplicated(keep="last")]
    common = new.index.intersection(old.index)
    a, b = new.loc[common].to_numpy(), old.loc[common].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(a - b) / np.abs(b)
    rel = np.where(np.isfinite(rel), rel, 0.0)
    worst = rel.max(axis=1) if rel.size else np.zeros(0)
    over = np.flatnonzero(worst > ADJUSTMENT_RTOL)
    return {
        "latest": latest.snapshot_id,
        "previous": previous.snapshot_id,
        "n_overlap": len(common),
        "max_rel_diff": float(worst.max()) if worst.size else 0.0,
        "discontinuous": bool(over.size),
        "first_time": _iso(common[over[0]]) if over.size else None,
    }


def _histogram(staleness: np.ndarray) -> dict[str, int]:
    values = np.asarray(staleness)
    counts = {
        str(int(v)): int(n) for v, n in zip(*np.unique(values[values >= 0], return_counts=True), strict=True)
    }
    never = int((values < 0).sum())
    if never:
        counts["no_value"] = never
    return counts


def build_quality_report(
    obs: ObservationSet,
    *,
    generations: tuple[ObservationSet, ObservationSet] | None = None,
    staleness: Mapping[str, np.ndarray] | None = None,
    dropped_rows: Mapping[str, int] | None = None,
) -> DataQualityReport:
    """Run all checks on the raw primary prices.

    Args:
        obs: the primary prices as fetched (duplicates and non-positive prices are allowed here).
        generations: the two newest cache generations of the primary request `(newest, previous)`,
            for the adjustment check (independent of which generation `obs` was served from).
        staleness: per covariate name, its staleness on the grid (`PrepDiagnostics.staleness`).
        dropped_rows: rows removed per reason (missing prices, warm-up, each covariate's policy).

    Returns:
        The report; empty lists mean "no finding".
    """
    return DataQualityReport(
        symbols=list(obs.series_names),
        n_rows=int(obs.values.shape[0]),
        gaps=_gaps(_times(obs), str(obs.meta.get("interval", "1d"))),
        duplicates=_duplicates(_times(obs)),
        non_positive_prices=_non_positive(obs),
        outliers=_outliers(obs),
        adjustment=None if generations is None else _adjustment(*generations),
        staleness={name: _histogram(s) for name, s in (staleness or {}).items()},
        dropped_rows=dict(dropped_rows or {}),
    )
