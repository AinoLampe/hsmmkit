# E10-T11 — Lazy optional imports, extras in `pyproject.toml`, import-time test, `extras` CI job

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | NFR-11, D22 | M8 | Part II §15.5, §16.2, §16.5 |

## Goal
Guarantee that `import hsmmkit` pulls only core dependencies and that every extra is installable and testable in isolation.

## Context (self-contained)
`hsmmkit.utils.optional_import(module: str, *, extra: str) -> ModuleType`: imports on first call, caches, on `ImportError` raises `ImportError(f"'{module}' is required for this feature: pip install 'hsmmkit[{extra}]'")` chained to the original. All modules using extras call it inside functions (never at module top level).
`pyproject.toml` extras:
```toml
[project.optional-dependencies]
gpu     = ["jax[cuda12]>=0.11"]
plot    = ["matplotlib>=3.9"]
sklearn = ["scikit-learn>=1.6"]
bayes   = ["numpyro>=0.20", "arviz>=0.21"]
mcmc    = ["numpyro>=0.20", "arviz>=0.21"]   # deprecated alias of bayes, removed in 0.5.0
xarray  = ["xarray>=2025.1"]
all     = ["hsmmkit[plot,sklearn,bayes,xarray]"]
```
Using the `mcmc` extra name has no runtime signal; document the deprecation in `docs/guide/deprecations.md` and `CHANGELOG`.
Import-time test (`tests/unit/test_e10_t11_import_time.py`): subprocess `python -X importtime -c "import hsmmkit"`; parse imported top-level module names; assert none of `matplotlib, sklearn, numpyro, arviz, xarray, scipy` (scipy until ADR-0020 decides), and assert total self-time below a recorded baseline × 1.3 (baseline JSON committed, refreshed only with justification).
Markers per extra: `plot`, `sklearn`, `bayes`, `xarray`; tests importing an extra are marked and skipped when it is missing (`pytest.importorskip` inside a fixture).
CI job `extras` (matrix over `[none, plot, sklearn, bayes, xarray]`): `uv sync --extra <x>` in a fresh env; run `pytest -m "<x> or not (plot or sklearn or bayes or xarray)"`; the `none` row asserts the actionable `ImportError` message for each extra.
The xarray source (E10-T04b) is switched to `optional_import("xarray", extra="xarray")`.

## Creates
- `src/hsmmkit/utils/optional.py`, `pyproject.toml` extras, `benchmarks/import_time_baseline.json`
- `tests/unit/test_e10_t11_import_time.py`, `tests/unit/test_e10_t11_optional_import.py`, `.github/workflows/pr.yml` (`extras` job)
- `docs/guide/install.md` (extras table), `docs/guide/deprecations.md` (created; `mcmc` alias entry)

## Acceptance criteria
1. `python -c "import hsmmkit"` in a core-only env imports none of the extras' packages (import-time test).
2. Each extra's tests run in isolation in the `extras` job; the `none` row shows the actionable message naming the extra.
3. `uv sync --extra mcmc` and `--extra bayes` resolve the same packages.

## Tests
Import-time test, optional-import unit tests, CI matrix.

## Out of scope
The features behind the extras (E13-T06a, E16-T01…T03).

## Connections
- Depends on: E10-T08
- Blocks: E13-T06a, E16-T01, E16-T02, E17-T02
- Relates to: E10-T04b, E14-T03 (scipy decision)
