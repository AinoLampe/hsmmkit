"""Shared pytest configuration.

Sets `REGIMEKIT_TYPECHECK=1` so `@jaxtyped(typechecker=beartype)` is active during tests, imports
`regimekit` before any JAX array is created (x64 must be enabled first, see `utils.jax_setup`,
E0-T05), and registers a `deadline=None` Hypothesis profile (first-call JIT compilation exceeds the
default deadline) and provides the `gpu_device` fixture (E4-T07).
"""

import importlib.abc
import importlib.machinery
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

os.environ.setdefault("REGIMEKIT_TYPECHECK", "1")

BLOCKED_ROOTS = ("regimekit.finance", "yfinance", "fredapi", "pandas_datareader", "curl_cffi")


class CoreOnlyFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that blocks the finance profile and the provider packages (`HSMM_CORE_ONLY=1`)."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Raise `ImportError("finance blocked")` for a blocked module, otherwise defer."""
        for root in BLOCKED_ROOTS:
            if fullname == root or fullname.startswith(root + "."):
                raise ImportError(f"finance blocked: {fullname}")
        return None


def install_core_only_finder() -> CoreOnlyFinder:
    """Put the blocking finder first on `sys.meta_path` and drop already imported blocked modules."""
    finder = CoreOnlyFinder()
    for name in [n for n in sys.modules if any(n == r or n.startswith(r + ".") for r in BLOCKED_ROOTS)]:
        del sys.modules[name]
    sys.meta_path.insert(0, finder)
    return finder


import jax  # noqa: E402
import pytest  # noqa: E402
from hypothesis import HealthCheck, settings  # noqa: E402

import regimekit  # noqa: E402

# Armed after `import regimekit`: until E10-T07b, `regimekit.project` still imports the finance providers
# (finance_import_allowlist.txt), so the package itself cannot be imported with the finder already active.
# `install_core_only_finder` drops the finance modules from `sys.modules`; later imports raise.
if os.environ.get("HSMM_CORE_ONLY") == "1":
    install_core_only_finder()

settings.register_profile(
    "regimekit",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("regimekit")

_ = regimekit.__version__  # touch the import so linters don't flag it as unused


@pytest.fixture(scope="session")
def gpu_device() -> "jax.Device":
    """The first JAX GPU device; skips (never fails) when no GPU backend exists.

    `jax.devices("gpu")` raises `RuntimeError` when the backend is absent (e.g. `JAX_PLATFORMS=cpu`)
    and returns an empty list on some builds; both mean "no GPU" (CLAUDE.md §5).
    """
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("no GPU backend available")
    return devices[0]


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Core-only runs skip modules that are marked `finance` as a whole (they would fail on import)."""
    if os.environ.get("HSMM_CORE_ONLY") != "1" or collection_path.suffix != ".py":
        return None
    if "finance" in collection_path.parts[-3:-1] or "live" in collection_path.parts[-3:-1]:
        return True
    try:
        head = collection_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return re.search(r"^pytestmark\s*=.*\bmark\.finance\b", head, re.MULTILINE) is not None
