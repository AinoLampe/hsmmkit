"""E10-T11: `optional_import`, extras in `pyproject.toml`, the deprecated `mcmc` alias."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from hsmmkit.utils.optional import optional_import

ROOT = Path(__file__).resolve().parents[2]
EXTRAS = {"plot": "matplotlib", "sklearn": "sklearn", "bayes": "numpyro", "xarray": "xarray"}


def test_e10_t11_optional_import_returns_the_module() -> None:
    """An installed module is returned (and is the cached `sys.modules` entry)."""
    assert optional_import("json", extra="none") is sys.modules["json"]


@pytest.mark.parametrize(("extra", "module"), sorted(EXTRAS.items()))
def test_e10_t11_missing_module_names_the_extra(
    extra: str, module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each extra's module missing gives the actionable message, chained to the original error."""
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError) as info:
        optional_import(module, extra=extra)
    assert str(info.value) == f"'{module}' is required for this feature: pip install 'hsmmkit[{extra}]'"
    assert isinstance(info.value.__cause__, ImportError)


def test_e10_t11_extras_table() -> None:
    """The extras of the ticket exist; `mcmc` resolves the same packages as `bayes` (AC3)."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert {"gpu", "plot", "sklearn", "bayes", "mcmc", "xarray", "all"} <= set(extras)
    assert extras["mcmc"] == extras["bayes"]
    assert extras["all"] == ["hsmmkit[plot,sklearn,bayes,xarray]"]
    assert any(d.startswith("arviz") for d in extras["bayes"])


def test_e10_t11_every_extra_has_a_marker() -> None:
    """A marker per extra lets the `extras` job select its tests."""
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = {m.partition(":")[0] for m in cfg["tool"]["pytest"]["ini_options"]["markers"]}
    assert {"plot", "sklearn", "bayes", "xarray"} <= markers


def test_e10_t11_no_extra_package_is_imported_at_module_level() -> None:
    """Modules of `src/` import an extra's package only inside functions."""
    banned = ("matplotlib", "sklearn", "numpyro", "arviz", "xarray")
    hits = []
    for path in (ROOT / "src").rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if line == stripped and stripped.startswith(("import ", "from ")):
                head = stripped.split()[1].split(".")[0]
                if head in banned and path.name != "xarray.py":
                    hits.append(f"{path.relative_to(ROOT)}:{n}")
    assert not hits, hits


def test_e10_t11_deprecations_page_lists_the_mcmc_alias() -> None:
    """`docs/guide/deprecations.md` documents the alias and its removal version."""
    text = (ROOT / "docs" / "guide" / "deprecations.md").read_text(encoding="utf-8")
    assert "hsmmkit[mcmc]" in text
    assert "0.5.0" in text
