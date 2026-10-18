# E0-T01 — Repository skeleton (uv project, src layout, package tree)

| Size | Requirements | Milestone | Concept refs |
|---|---|---|---|
| S | D11, NFR-06, NFR-08 | M0 | §5.3, §6.1 |

> **Amended 2026-09-14** — concept v1.1, Part II §15.7 (amendment 3: `finance/` subpackage in the initial tree). No behavioural change for the finance profile.

## Goal
Create the installable, empty-but-fully-structured `regimekit` package with all extras and dependency groups resolvable by `uv`.

## Context (self-contained)
`pyproject.toml` baseline (extend, do not shrink):

```toml
[project]
name = "regimekit"; requires-python = ">=3.13"; license = "Apache-2.0"
dependencies = ["jax>=0.11", "equinox>=0.13", "optax>=0.2.8", "numpy>=2.1", "pandas>=2.2",
  "pyarrow>=25", "pydantic>=2.13", "ruamel.yaml>=0.18", "typer>=0.15", "structlog>=25", "jaxtyping>=0.3"]
[project.optional-dependencies]
gpu = ["jax[cuda12]>=0.11"]
providers = ["yfinance>=1.7", "curl_cffi>=0.15", "pandas-datareader>=0.10", "fredapi>=0.5"]
mcmc = ["numpyro>=0.20"]
[dependency-groups]
dev = ["pytest>=9", "pytest-cov", "pytest-xdist", "pytest-benchmark", "hypothesis>=6.163", "beartype",
  "ruff>=0.16", "mypy>=2.1", "pre-commit", "nox", "scipy>=1.15", "mkdocs-material>=9.7", "mkdocstrings[python]"]
[project.scripts]
regimekit = "regimekit.cli.app:main"
```
Package tree to create as empty packages (each with `__init__.py` and a one-line module docstring): see overview §4.2 — this includes `src/regimekit/finance/` and `src/regimekit/finance/providers/` (the finance profile; ADR-0012). `data/providers/` does **not** exist. Extras stay `gpu`, `providers`, `mcmc`. `src/regimekit/__init__.py` exposes `__version__` (read from package metadata) only; the JAX bootstrap import and `Project` re-export are added by E0-T05 and E8-T02. No provider data is ever committed (`.gitignore`: `runs/`, `*.parquet`, `.cache/`).

## Creates
- `pyproject.toml`, `uv.lock`, `README.md` (stub), `LICENSE` (Apache-2.0), `.gitignore`, `.python-version`
- `src/regimekit/**/__init__.py` for every package in overview §4.2
- `tests/` skeleton directories with `conftest.py` (empty) and `tests/fixtures/README.md`
- `docs/`, `examples/` placeholder directories

## Acceptance criteria
1. `uv sync --all-groups` succeeds on Linux and WSL2.
2. `python -c "import regimekit; print(regimekit.__version__)"` works.
3. `uv sync --extra gpu` resolves (installation of CUDA wheels not required in CI); `[tool.uv.sources]` documented if needed.
4. `uv build` produces sdist + wheel containing only `src/regimekit`.

## Tests
- `tests/unit/test_e0_t01_import.py`: imports every subpackage listed in overview §4.2, including `regimekit.finance` and `regimekit.finance.providers`.

## Out of scope
Ruff/mypy/pytest configuration (E0-T02), CI (E0-T03), any real code.

## Connections
- Depends on: —
- Blocks: E0-T02, E0-T04, E0-T05
- Relates to: ADR-0002 (license), ADR-0003 (stack)
