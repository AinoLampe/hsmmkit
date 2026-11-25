"""E7-T02b: `evaluation.backtest.n_sim` config field (CLAUDE.md 3.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from regimekit.config.schema import BacktestCfg
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.finance import FinanceConfig

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "configs"
_YAML = YAML(typ="safe")


def test_e7_t02b_n_sim_defaults_to_none_validates_and_is_in_schema_and_resolved_dump(tmp_path: Path) -> None:
    """`n_sim` is optional (None = `forecast.n_sim`), must be >= 1, and appears in the schema and the dump."""
    base = {"scheme": "expanding", "initial_train": 2.0, "refit_every_steps": 63, "horizons": [1]}
    assert BacktestCfg(**base).n_sim is None  # type: ignore[arg-type]
    assert BacktestCfg(**base, n_sim=500).n_sim == 500  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BacktestCfg(**base, n_sim=0)  # type: ignore[arg-type]

    schema_path = tmp_path / "schema.json"
    export_json_schema(schema_path, FinanceConfig)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prop = schema["$defs"]["BacktestCfg"]["properties"]["n_sim"]
    assert prop["default"] is None
    assert {"type": "integer", "minimum": 1} in prop["anyOf"]

    resolved = tmp_path / "config.resolved.yaml"
    dump_resolved(load_config(FIXTURES / "appendix_a.yaml", FinanceConfig), resolved)
    assert _YAML.load(resolved)["evaluation"]["backtest"]["n_sim"] is None
