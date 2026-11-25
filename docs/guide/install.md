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
| `mcmc` | `numpyro` (reserved for Bayesian extensions) | `uv sync --extra mcmc` |

The `csv`, `parquet`, `numpy` and `pandas` sources need no extra.

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
