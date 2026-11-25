"""Lazy imports of optional dependencies (E10-T04b, E10-T11).

Every module that needs an extra (`plot`, `sklearn`, `bayes`, `xarray`) calls `optional_import` inside a
function, never at module top level, so that `import hsmmkit` pulls only the core dependencies.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["optional_import"]


def optional_import(module: str, *, extra: str) -> ModuleType:
    """Import `module` (cached by `sys.modules`), or raise an `ImportError` that names the extra to install.

    Raises:
        ImportError: `module` is not installed: `'<module>' is required for this feature: pip install
            'hsmmkit[<extra>]'`, chained to the original error.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"'{module}' is required for this feature: pip install 'hsmmkit[{extra}]'") from exc
