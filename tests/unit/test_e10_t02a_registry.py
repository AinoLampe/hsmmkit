"""E10-T02a: component registries (FR-G05, D15, D24)."""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from regimekit.plugins import KINDS, REGISTRY, Registry, provenance_of, register
from regimekit.utils.errors import ConfigError

SRC = Path(__file__).resolve().parents[2] / "src" / "regimekit"


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
    reg.register("metric", "m", 1, provenance="regimekit.core")
    with pytest.raises(ConfigError, match=r"regimekit\.core.*ext\.pkg|ext\.pkg.*regimekit\.core"):
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
    """The decorator labels components registered from core, finance and extension modules."""
    assert provenance_of("regimekit.model.emission.gbm") == "regimekit.core"
    assert provenance_of("regimekit.finance.providers.csv") == "regimekit.finance"
    assert provenance_of("regimekit.finance") == "regimekit.finance"
    assert provenance_of("myext.emission.foo") == "myext"
    assert REGISTRY.entries("emission")[0].provenance == "regimekit.core"


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


def test_e10_t02a_part_one_components_are_registered() -> None:
    """Every Part I family, init strategy and forecaster resolves through the registry."""
    import regimekit.evaluation.metrics.registry as metrics
    import regimekit.inference.covariate_forecasters
    import regimekit.model.build  # noqa: F401

    assert set(REGISTRY.names("emission")) == {"gbm", "product", "student_t"}
    assert set(REGISTRY.names("duration")) == {
        "gamma",
        "geometric",
        "lognormal",
        "negbin",
        "nonparametric",
        "poisson",
        "weibull",
    }
    assert set(REGISTRY.names("init")) == {"lattice_jitter", "random", "rollvol_kmeans"}
    assert set(REGISTRY.names("covariate_forecaster")) == {"ar", "bootstrap", "freeze", "scenario"}
    assert set(REGISTRY.names("metric")) == set(metrics.available())
    assert REGISTRY.names("clock") == REGISTRY.names("transform") == REGISTRY.names("preset") == ()


def test_e10_t02a_config_resolves_every_part_one_family_through_the_registry() -> None:
    """The config -> ModelSpec lookups are views over the registry (no private dicts)."""
    from regimekit.model.build import DURATION_FAMILIES, EMISSION_FAMILIES

    assert not isinstance(EMISSION_FAMILIES, dict)
    assert not isinstance(DURATION_FAMILIES, dict)
    assert set(EMISSION_FAMILIES) == {"gbm", "student_t"}  # `product` has no config name
    assert set(DURATION_FAMILIES) == set(REGISTRY.names("duration"))
    for name in DURATION_FAMILIES:
        assert DURATION_FAMILIES[name] is REGISTRY.get("duration", name)
    assert EMISSION_FAMILIES.get("nope") is None


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


def test_e10_t02a_core_only_registry_has_no_finance_emissions() -> None:
    """AC4: with HSMM_CORE_ONLY=1 no emission has provenance regimekit.finance (vacuous until E10-T06)."""
    code = (
        "import sys; sys.path.insert(0, 'tests'); import conftest\n"
        "from regimekit.plugins import REGISTRY\n"
        "import regimekit.model.build\n"
        "print(sorted({e.provenance for e in REGISTRY.entries('emission')}))\n"
    )
    root = SRC.parents[1]
    env = dict(os.environ, HSMM_CORE_ONLY="1")
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "['regimekit.core']"


def test_e10_t02a_metric_facade_keeps_its_messages_and_replace() -> None:
    """Metric facade keeps its messages and replace."""
    from regimekit.evaluation.metrics import registry as m

    names = m.available()
    assert names == sorted(names)
    first = m.resolve([names[0]])[0]
    with pytest.raises(ConfigError, match="already registered"):
        m.register(first)
    assert m.register(first, replace=True) is first
    with pytest.raises(ConfigError, match=r"unknown metric\(s\) \['nope'\]; known:"):
        m.resolve(["nope"])
    importlib.import_module("regimekit.evaluation.metrics.registry")
