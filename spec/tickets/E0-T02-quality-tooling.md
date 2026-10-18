# E0-T02 — Quality tooling (ruff, mypy strict, coverage gate, pytest markers, pre-commit, nox)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | NFR-04, NFR-05 | M0 | §6.1, §6.2, §6.5 |

## Goal
Make the quality gates from the concept executable locally via `nox` and `pre-commit` before any real code exists.

## Context (self-contained)
```toml
[tool.coverage.report]
fail_under = 90; show_missing = true; exclude_also = ["if TYPE_CHECKING:", "@overload"]
[tool.mypy]
strict = true; plugins = ["pydantic.mypy"]; files = ["src"]
[tool.ruff]
line-length = 110; target-version = "py313"
[tool.ruff.lint]
select = ["E","F","W","I","N","UP","B","SIM","PT","RUF","D","ANN","NPY"]; pydocstyle.convention = "google"
[tool.pytest.ini_options]
markers = ["slow: nightly suites", "live: network tests", "benchmark: pytest-benchmark", "gpu: requires GPU"]
addopts = "-m 'not slow and not live and not benchmark and not gpu'"
```
Env `REGIMEKIT_TYPECHECK=1` must be set in the `tests` session so `@jaxtyped(typechecker=beartype)` is active. Coverage per package ≥ 85 % (`model, engine, estimation, inference, data, evaluation`): implement as a small script `scripts/check_package_coverage.py` reading `coverage.json`. Provider network-call lines may carry `# pragma: no cover`.

nox sessions: `lint` (ruff check + format --check), `types` (mypy), `tests` (`pytest -n auto --cov`), `recovery` (`-m slow`), `docs` (`mkdocs build --strict`), `bench` (`-m benchmark`).
pre-commit hooks: ruff format, ruff lint, mypy (staged files), end-of-file-fixer, check-yaml, `uv lock --check`.

## Creates
- `pyproject.toml` tool sections above; `noxfile.py`; `.pre-commit-config.yaml`; `scripts/check_package_coverage.py`
- `tests/conftest.py`: sets `REGIMEKIT_TYPECHECK=1`, registers markers

## Acceptance criteria
1. `nox -s lint types tests` green on the E0-T01 skeleton.
2. A deliberately wrong annotation in a throwaway module is caught by `nox -s types` (test via a temporary file in CI job or documented manual check; a unit test may run mypy on a fixture file).
3. `pre-commit run --all-files` passes.
4. Coverage config produces `coverage.json`; per-package script fails when a package is below 85 %.

## Tests
- `tests/unit/test_e0_t02_markers.py`: markers are registered; default run excludes `slow/live/benchmark/gpu`.

## Out of scope
GitHub Actions (E0-T03); docs content (E0-T04).

## Connections
- Depends on: E0-T01
- Blocks: E0-T03
- Relates to: overview §4.6–4.8
