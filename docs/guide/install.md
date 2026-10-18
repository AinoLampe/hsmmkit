# Install

regimekit needs Python 3.13 or newer and is installed with [uv](https://docs.astral.sh/uv/) (pip is not
supported for development). JAX runs on the CPU by default.

## From source

```bash
git clone <repository-url> regimekit && cd regimekit
uv sync --all-groups        # runtime + dev tools (pytest, ruff, mypy, mkdocs, ...)
uv run regimekit --help
```

For use without the dev tools: `uv sync` installs only the runtime dependencies.

## Optional extras

| Extra | Adds | Install |
|---|---|---|
| `providers` | `yfinance`, `curl_cffi`, `pandas-datareader`, `fredapi` (the network data providers) | `uv sync --extra providers` |
| `gpu` | `jax[cuda12]` | `uv sync --extra gpu` (CUDA 12, Linux) |
| `mcmc` | `numpyro` (reserved for Bayesian extensions) | `uv sync --extra mcmc` |

The `csv` provider needs no extra: it reads local files, which is how the tutorial and CI run offline.
FRED series need a `FRED_API_KEY` environment variable; keys are read from the environment only and never
written to configs, caches or logs (see the [providers guide](providers.md)).

## GPU

`estimation.device: auto` uses a GPU when JAX finds one and the CPU otherwise; `--device gpu` without a GPU
backend fails with a `NumericalError` (exit code 4). CUDA wheels exist for Linux only: on Windows use WSL2.
GPU tests are marked `gpu` and are not part of the default test run.

## Verify the installation

```bash
uv run regimekit selftest
```

fits simulated data twice on your backend in about a minute and exits non-zero unless the fits are finite
and bitwise reproducible (see [Validation](validation.md)).
