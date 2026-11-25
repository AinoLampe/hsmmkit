"""Lazy imports of optional dependencies (E10-T04b; the extras themselves arrive with E10-T11)."""

from __future__ import annotations

import importlib
from types import ModuleType


def optional_import(module: str, extra: str) -> ModuleType:
    """Import `module`, or raise an `ImportError` that names the extra to install.

    Raises:
        ImportError: `module` is not installed (`pip install hsmmkit[<extra>]`).
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"pip install hsmmkit[{extra}]  (needed for {module})") from exc
