"""E10-T10: feature gates (Part II overview §5.11): a gated key validates, but the model build rejects it."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from hsmmkit.config import gates
from hsmmkit.config.schema import parse_project_config
from hsmmkit.model.build import BuildInputs, build_model_spec_from
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw
from tests.unit.test_e10_t10_schema import appendix_h

INPUTS = BuildInputs(n_series=3, cov_names={}, clock="steps", nominal_dt=1.0)


def _explicit(raw: dict[str, Any]) -> dict[str, Any]:
    raw["model"].update(
        engine="explicit",
        d_max=60,
        duration={
            "family": "negbin",
            "r": [1, 1],
            "params": {"mean": {"mode": "free", "init": [10.0, 20.0]}},
        },
    )
    return raw


# One mutation per `GATES` row, applied to the Part I core config (which builds): the row must fire.
def _case(name: str) -> dict[str, Any]:
    raw = core_raw(Path("/tmp/x"))
    if name == "model.left_censored":
        raw = _explicit(raw)
        raw["model"]["left_censored"] = True
    elif name == "model.duration.per_state":
        raw = _explicit(raw)
        raw["model"]["duration"] = {"per_state": [{"family": "geometric"}, {"family": "geometric"}]}
    elif name == "model.duration.duration_dependent_jumps":
        raw = _explicit(raw)
        raw["model"]["duration"]["duration_dependent_jumps"] = True
    elif name == "data.labels":
        raw["data"]["labels"] = {"source": "column", "column": "y"}
    elif name == "estimation.init":
        raw["data"]["labels"] = {"source": "column", "column": "y"}
        raw["estimation"]["init"] = ["labels"]
    elif name == "model.emission.covariance":
        raw["model"]["emission"]["covariance"] = "full"
    elif name.startswith("estimation.method="):
        raw["estimation"]["method"] = name.partition("=")[2]
    elif name == "evaluation.cv":
        raw["evaluation"]["cv"] = {"splitter": "sequence_kfold", "folds": 3}
    elif name == "evaluation.grid":
        raw["evaluation"]["grid"] = {"n_states": [2, 3]}
    elif name == "evaluation.metrics":
        raw["evaluation"]["metrics"] = ["log_score", "ari"]
    elif name == "online":
        raw["online"] = {}
    elif name == "online.lag":
        raw["online"] = {"lag": 5}
    elif name == "online.alerts":
        raw["online"] = {"alerts": [{"regime": 1, "p_on": 0.8, "p_off": 0.5}]}
    else:  # pragma: no cover - a new row needs a case here
        raise AssertionError(f"no case for gate {name}")
    return raw


def test_e10_t10_part1_config_passes_every_gate(tmp_path: Path) -> None:
    """A Part I-derived core config uses no gated feature and builds."""
    cfg = parse_project_config(core_raw(tmp_path))
    assert gates.used_gates(cfg) == []
    gates.check(cfg)


@pytest.mark.parametrize("row", list(gates.GATES))
def test_e10_t10_each_gate_row_fires_with_its_ticket(row: str) -> None:
    """Every row of `GATES` rejects its key at build with `not available before <ticket>`, path = the key."""
    cfg = parse_project_config(_case(row))
    assert row in dict(gates.used_gates(cfg))
    first_path, first_ticket = gates.used_gates(cfg)[0]
    with pytest.raises(ConfigError) as info:
        gates.check(cfg)
    assert info.value.path == first_path.partition("=")[0]
    assert f"not available before {first_ticket}" in str(info.value)


def test_e10_t10_validation_is_not_gated(tmp_path: Path) -> None:
    """Gated keys validate (presets and the JSON Schema accept them); only the build refuses."""
    for row in gates.GATES:
        parse_project_config(_case(row))


def test_e10_t10_appendix_h_build_raises_the_first_gate_in_deterministic_order() -> None:
    """Appendix H builds into `data.labels` -> E12-T06 first (the order of `GATES`)."""
    cfg = parse_project_config(appendix_h())
    paths = [p for p, _ in gates.used_gates(cfg)]
    assert paths[0] == "data.labels"
    assert paths == [p for p in gates.GATES if p in paths]  # GATES order
    with pytest.raises(ConfigError) as info:
        build_model_spec_from(cfg, INPUTS)
    assert info.value.path == "data.labels"
    assert "E12-T06" in str(info.value)


def test_e10_t10_removing_a_row_lets_the_key_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mechanism a feature ticket uses: delete its row, and the key no longer raises."""
    cfg = parse_project_config(_case("data.labels"))
    monkeypatch.setattr(gates, "GATES", {k: v for k, v in gates.GATES.items() if k != "data.labels"})
    gates.check(cfg)


def test_e10_t10_gates_do_not_mutate_the_config() -> None:
    """`check` only reads."""
    cfg = parse_project_config(appendix_h())
    before = copy.deepcopy(cfg.model_dump())
    with pytest.raises(ConfigError):
        gates.check(cfg)
    assert cfg.model_dump() == before
