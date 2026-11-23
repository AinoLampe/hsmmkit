# E17-T02 — Distribution: PyPI trusted publishing, conda-forge feedstock, extras smoke installs

| Repo | Size | Requirements | Milestone | Concept refs |
|---|---|---|---|---|
| hsmmkit | S | D22, NFR-06, NFR-11 | M13 | Part II §15.5, §16.5; Part I E8-T04 (release workflow) |

## Goal
Make `pip install hsmmkit[...]` and `conda install -c conda-forge hsmmkit` work from public indexes, verified automatically.

## Context (self-contained)
- **PyPI**: GitHub Actions release workflow (`release.yml`, from E8-T04/E10-T08) gains a `pypi` job using PyPI **trusted publishing** (OIDC, `pypa/gh-action-pypi-publish`, environment `pypi` with required reviewer), triggered on tags `v*`; TestPyPI job on pre-release tags `v*rc*`. Artifacts: sdist + pure-Python wheel built with `uv build`; `twine check` in CI.
- **Project metadata**: classifiers (Python 3.13+, OS independent, Science/Research, Apache-2.0), `project.urls` (docs, source, changelog), long description from README (domain-neutral, with a short "used by regimekit" note).
- **conda-forge**: prepare `recipe/meta.yaml` (noarch: python; run deps mapping incl. `jax`, `jaxlib`, `equinox`, `optax`; extras as separate `hsmmkit-<extra>` outputs or documented `pip` fallback — decision recorded in the recipe README) and submit to `staged-recipes`; this ticket is done when the PR is opened and the recipe builds in its CI; the feedstock merge is tracked in E17-T04.
- **Smoke installs** (CI matrix nightly and on release tags): fresh environments on Linux (ubuntu-latest), WSL2-equivalent (Ubuntu container), macOS (CPU, best effort) × install variants `hsmmkit`, `hsmmkit[plot,sklearn,xarray]`, `hsmmkit[bayes]`; each runs `hsmmkit selftest --quick` and the import-time test.
- **`RegimeDetectionAndForecast`**: unaffected here (it consumes `hsmmkit` from PyPI from E17-T05 on).

## Creates
- `.github/workflows/release.yml` (pypi/testpypi jobs), `.github/workflows/smoke-install.yml`, `recipe/meta.yaml`, `recipe/README.md`
- `pyproject.toml` metadata updates

## Acceptance criteria
1. A release-candidate tag publishes to TestPyPI via trusted publishing; `pip install -i https://test.pypi.org/simple hsmmkit==<rc>` works in a fresh env.
2. Smoke-install matrix green on Linux; macOS failures are reported but non-blocking (documented).
3. conda-forge staged-recipes PR opened and its CI green.

## Tests
CI workflows themselves; `twine check`.

## Out of scope
Publishing the final 0.5.0 (E17-T04).

## Connections
- Depends on: E10-T11
- Blocks: E17-T04
- Relates to: E8-T04, E10-T08
