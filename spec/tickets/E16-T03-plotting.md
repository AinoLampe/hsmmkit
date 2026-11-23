# E16-T03 — `hsmmkit.plot` (matplotlib, `plot` extra)

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | M | FR-VZ01, FR-VZ02, D20 | M12 | Part II §13.1 FR-VZ, §15.3 (`Plotter`) |

## Goal
Provide a small set of side-effect-free plotting helpers for every result type.

## Context (self-contained)
All helpers: signature `fn(result, ax: Axes | None = None, **kw) -> Axes`; create a figure only when `ax is None` (and never call `plt.show()`); no global rcParams changes (use `with plt.rc_context(...)` locally if needed); colors from a fixed qualitative palette indexed by state (colour-blind safe, documented) — overridable via `colors=`. Matplotlib via `optional_import("matplotlib", extra="plot")` and the `Agg` backend in tests.
| Module / function | Content |
|---|---|
| `plot.regimes.observations(post, data, series=0, ax)` | observation line with background shading by Viterbi regime (alpha by smoothed probability optional) |
| `plot.regimes.probabilities(post, kind="smoothed"|"filtered", ax)` | stacked area of regime probabilities over time |
| `plot.regimes.segments(post, ax)` | horizontal bars per segment (Gantt-like), FFBS uncertainty band if `path_samples` exist (E13-T04) |
| `plot.durations.pmf(fit, state=None, ax)` / `hazard(...)` | duration pmfs/hazards from the families (`log_pmf`, `hazard`) in clock units |
| `plot.forecast.fan(fc, series=0, quantiles=None, ax)` | fan chart of cumulative quantiles with history tail |
| `plot.diagnostics.pit(fit_or_post, ax)` | PIT histogram with uniform band |
| `plot.diagnostics.qq(residuals, ax)` | normal QQ of pseudo-residuals (E14-T05) |
| `plot.diagnostics.restarts(fit, ax)` | restart objective dispersion (dot plot) |
| `plot.diagnostics.trace(fit, ax)` | ELBO trace (VB) or EM loglik trace; for MCMC delegates to `arviz.plot_trace` if available (returns its axes) |
| `plot.diagnostics.panel(fit, post, residuals=None)` | 2×2 figure composed of the above; returns `Figure` (documented exception to the `Axes` rule) |
Batched results: `sequence=` argument selects the sequence (default first).

## Creates
- `src/hsmmkit/plot/{__init__,regimes,durations,forecast,diagnostics}.py`
- `tests/unit/test_e16_t03_plots.py` (marker `plot`), image smoke tests `tests/images/` (baseline PNGs, tolerance-based comparison with `matplotlib.testing.compare.compare_images`, tol 10)
- `docs/guide/core/plotting.md` (gallery generated in the docs build)

## Acceptance criteria
1. Every helper renders headless with `Agg`, returns `Axes` (or `Figure` for `panel`), creates no figure when `ax` is passed, and leaves `plt.get_fignums()` unchanged in that case.
2. Image smoke tests pass within tolerance; baselines regenerated only via a documented script.
3. Importing `hsmmkit` does not import matplotlib (E10-T11 test still green).

## Tests
Unit and image-smoke tests.

## Out of scope
Interactive plots (plotly) — plug-in example only (Part II §19 #16).

## Connections
- Depends on: E16-T02
- Blocks: E16-T05, E17-T03
- Relates to: E13-T04, E14-T05
