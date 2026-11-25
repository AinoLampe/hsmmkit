# Install

hsmmkit needs Python 3.13 or newer and is installed with [uv](https://docs.astral.sh/uv/) (pip is not
supported for development). JAX runs on the CPU by default.

## From source

```bash
git clone <repository-url> hsmmkit && cd hsmmkit
uv sync --all-groups        # runtime + dev tools (pytest, ruff, mypy, mkdocs, ...)
uv run hsmmkit --help
```

For use without the dev tools: `uv sync` installs only the runtime dependencies.

## Optional extras

| Extra | Adds | Install |
|---|---|---|
| `gpu` | `jax[cuda12]` | `uv sync --extra gpu` (CUDA 12, Linux) |
| `plot` | `matplotlib` | `uv sync --extra plot` |
| `sklearn` | `scikit-learn` | `uv sync --extra sklearn` |
| `bayes` | `numpyro`, `arviz` | `uv sync --extra bayes` |
| `mcmc` | same as `bayes` | deprecated alias, removed in 0.5.0 ([deprecations](deprecations.md)) |
| `xarray` | `xarray` | `uv sync --extra xarray` (the `xarray` source) |
| `all` | `plot`, `sklearn`, `bayes`, `xarray` | `uv sync --extra all` |

The `csv`, `parquet`, `numpy` and `pandas` sources need no extra. A feature whose extra is missing raises an
`ImportError` that names it (`'xarray' is required for this feature: pip install 'hsmmkit[xarray]'`);
`import hsmmkit` itself imports none of the extras' packages.

## GPU

`estimation.device: auto` uses a GPU when JAX finds one and the CPU otherwise; `--device gpu` without a GPU
backend fails with a `NumericalError` (exit code 4). CUDA wheels exist for Linux only: on Windows use WSL2.
GPU tests are marked `gpu` and are not part of the default test run.

## Verify the installation

```bash
uv run hsmmkit selftest
```

fits simulated data twice on your backend in about a minute and exits non-zero unless the fits are finite
and bitwise reproducible (see [Validation](validation.md)).
