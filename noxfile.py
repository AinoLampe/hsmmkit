"""Nox sessions for regimekit's quality gates.

Sessions run inside the project's own uv-managed environment (`uv run ...`) rather than
nox-managed virtualenvs, so `uv sync --all-groups` is the single source of truth for
dependencies.
"""

import os

import nox

nox.options.default_venv_backend = "none"
nox.options.reuse_existing_virtualenvs = True


@nox.session
def lint(session: nox.Session) -> None:
    """Ruff lint + format check."""
    session.run("uv", "run", "ruff", "check", ".", external=True)
    session.run("uv", "run", "ruff", "format", "--check", ".", external=True)


@nox.session
def types(session: nox.Session) -> None:
    """Mypy strict on src/."""
    session.run("uv", "run", "mypy", external=True)


@nox.session
def tests(session: nox.Session) -> None:
    """Fast test suite with coverage (excludes slow/live/benchmark/gpu)."""
    env = dict(os.environ, REGIMEKIT_TYPECHECK="1", JAX_PLATFORMS="cpu")
    session.run(
        "uv",
        "run",
        "pytest",
        "-n",
        "auto",
        "--cov",
        "--cov-report=json",
        "--cov-report=term-missing",
        env=env,
        external=True,
    )
    session.run("uv", "run", "python", "scripts/check_package_coverage.py", env=env, external=True)


@nox.session
def recovery(session: nox.Session) -> None:
    """Full parameter-recovery suite (slow: six scenarios, ~4 h on a laptop; the weekly CI job)."""
    env = dict(os.environ, REGIMEKIT_TYPECHECK="1", JAX_PLATFORMS="cpu")
    session.run(
        "uv",
        "run",
        "pytest",
        "-m",
        "slow",
        "--ignore=tests/recovery/test_e5_t05a_recovery_smoke.py",
        "--ignore=tests/integration/test_e8_t03_notebook.py",
        env=env,
        external=True,
    )


@nox.session
def recovery_smoke(session: nox.Session) -> None:
    """Quick recovery smoke run at T=5000 (one scenario, a few minutes; the nightly CI job)."""
    env = dict(os.environ, REGIMEKIT_TYPECHECK="1", JAX_PLATFORMS="cpu")
    session.run(
        "uv",
        "run",
        "pytest",
        "-m",
        "slow",
        "tests/recovery/test_e5_t05a_recovery_smoke.py",
        env=env,
        external=True,
    )


@nox.session
def examples(session: nox.Session) -> None:
    """Execute the tutorial notebook offline on synthetic data (a few minutes)."""
    env = dict(os.environ, REGIMEKIT_TYPECHECK="1", JAX_PLATFORMS="cpu")
    session.run(
        "uv",
        "run",
        "pytest",
        "-m",
        "slow",
        "tests/integration/test_e8_t03_notebook.py",
        env=env,
        external=True,
    )


@nox.session
def docs(session: nox.Session) -> None:
    """Strict mkdocs build."""
    session.run("uv", "run", "mkdocs", "build", "--strict", external=True)


@nox.session
def bench(session: nox.Session) -> None:
    """Benchmark suite."""
    env = dict(os.environ, REGIMEKIT_TYPECHECK="1", JAX_PLATFORMS="cpu")
    session.run("uv", "run", "pytest", "-m", "benchmark", env=env, external=True)
