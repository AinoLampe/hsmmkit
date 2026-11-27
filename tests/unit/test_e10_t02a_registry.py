"""E10-T02a: component registries (FR-G05, D15, D24)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from hsmmkit.plugins import KINDS, REGISTRY, Registry, provenance_of, register
from hsmmkit.utils.errors import ConfigError

SRC = Path(__file__).resolve().parents[2] / "src" / "hsmmkit"


def test_e10_t02a_register_get_names_entries() -> None:
    """Register, look up, list sorted names and entries with provenance."""
    reg = Registry()
    reg.register("emission", "b", object, provenance="pkg.b")
    reg.register("emission", "a", int, provenance="pkg.a")
    assert reg.get("emission", "a") is int
    assert reg.names("emission") == ("a", "b")
    assert [(e.name, e.provenance) for e in reg.entries("emission")] == [("a", "pkg.a"), ("b", "pkg.b")]
    assert reg.names("duration") == ()


def test_e10_t02a_duplicate_registration_names_both_provenances() -> None:
    """AC1: a duplicate raises ConfigError naming both provenances."""
    reg = Registry()
    reg.register("metric", "m", 1, provenance="hsmmkit.core")
    with pytest.raises(ConfigError, match=r"hsmmkit\.core.*ext\.pkg|ext\.pkg.*hsmmkit\.core"):
        reg.register("metric", "m", 2, provenance="ext.pkg")
    assert reg.get("metric", "m") == 1


def test_e10_t02a_replace_overwrites() -> None:
    """Replace overwrites."""
    reg = Registry()
    reg.register("metric", "m", 1, provenance="a")
    reg.register("metric", "m", 2, provenance="b", replace=True)
    assert reg.get("metric", "m") == 2


def test_e10_t02a_unknown_name_lists_all_registered_names_with_provenance() -> None:
    """AC1: an unknown name lists all registered names with provenance."""
    reg = Registry()
    reg.register("duration", "x", 1, provenance="pkg.x")
    reg.register("duration", "y", 2, provenance="pkg.y")
    with pytest.raises(ConfigError) as err:
        reg.get("duration", "zzz")
    assert "unknown duration 'zzz'; registered: x (pkg.x), y (pkg.y)" in str(err.value)
    assert err.value.path is None


def test_e10_t02a_unknown_kind_is_a_config_error() -> None:
    """Unknown kind is a config error."""
    with pytest.raises(ConfigError, match="unknown registry kind"):
        Registry().names("nope")  # type: ignore[arg-type]


def test_e10_t02a_all_kinds_exist_and_clock_transform_preset_start_empty_or_registered() -> None:
    """All kinds exist and clock transform preset start empty or registered."""
    reg = Registry()
    assert set(KINDS) == {
        "estimator",
        "splitter",
        "emission",
        "duration",
        "metric",
        "source",
        "clock",
        "transform",
        "init",
        "covariate_forecaster",
        "preset",
    }
    assert all(reg.names(k) == () for k in KINDS)  # type: ignore[arg-type]


def test_e10_t02a_decorator_default_provenance_is_the_callers_origin() -> None:
    """The decorator labels components registered from core and extension modules."""
    assert provenance_of("hsmmkit.model.emission.gaussian") == "hsmmkit.core"
    assert provenance_of("hsmmkit.data.sources.csv") == "hsmmkit.core"
    assert provenance_of("regimekit.finance") == "regimekit"
    assert provenance_of("myext.emission.foo") == "myext"
    assert REGISTRY.entries("emission")[0].provenance == "hsmmkit.core"


def test_e10_t02a_decorator_registers_and_returns_the_object() -> None:
    """Decorator registers and returns the object."""
    reg_before = REGISTRY.names("preset")

    @register("preset", "__e10_t02a_probe__", provenance="test")
    def probe() -> int:
        return 1

    try:
        assert probe() == 1
        assert REGISTRY.get("preset", "__e10_t02a_probe__") is probe
    finally:
        REGISTRY._entries["preset"].pop("__e10_t02a_probe__")
    assert REGISTRY.names("preset") == reg_before


def _is_dict_value(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Dict) or (
        isinstance(value, ast.Call) and getattr(value.func, "id", "") == "dict"
    )


def test_e10_t02a_no_module_level_name_dicts_remain() -> None:
    """AC2: no module-level dict maps family/metric/init/forecaster names any more."""
    banned = {
        "_DATA_STRATEGIES",
        "_SPEC_ONLY_STRATEGIES",
        "_REGISTRY",
        "EMISSION_FAMILIES",
        "DURATION_FAMILIES",
    }
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "plugins.py" or "providers" in path.parts:  # providers move in E10-T04b
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            offenders += [
                f"{path.relative_to(SRC)}:{t.id}"
                for t in targets
                if isinstance(t, ast.Name) and t.id in banned and _is_dict_value(node.value)
            ]
    assert not offenders, offenders


def test_e10_t02a_every_registered_emission_is_a_core_component() -> None:
    """AC4: no emission in the registry has an application provenance."""
    import hsmmkit.model.build  # noqa: F401
    from hsmmkit.plugins import REGISTRY

    assert {e.provenance for e in REGISTRY.entries("emission")} == {"hsmmkit.core"}


def test_e10_t02a_metric_facade_keeps_its_messages_and_replace() -> None:
    """Metric facade keeps its messages and replace."""
    from hsmmkit.evaluation.metrics import registry as m

    names = m.available()
    assert names == sorted(names)
    first = m.resolve([names[0]])[0]
    with pytest.raises(ConfigError, match="already registered"):
        m.register(first)
    assert m.register(first, replace=True) is first
    with pytest.raises(ConfigError, match=r"unknown metric\(s\) \['nope'\]; known:"):
        m.resolve(["nope"])
    importlib.import_module("hsmmkit.evaluation.metrics.registry")
