"""E10-T10: preset registry and CLI, JSON-Schema export with registry enums."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pandas as pd
import pytest
from typer.testing import CliRunner

from hsmmkit.cli.app import app
from hsmmkit.config.schema import ProjectConfig
from hsmmkit.config.yaml_io import _SAFE_YAML, json_schema, load_config
from hsmmkit.presets import get_preset, list_presets
from hsmmkit.utils.errors import ConfigError

runner = CliRunner()


def test_e10_t10_core_presets_are_registered() -> None:
    """The two core presets are listed with their provenance in the registry kind `preset`."""
    names = [p.name for p in list_presets()]
    assert names == ["core/segmentation_gaussian_ltr", "core/two_regime_gaussian"]
    with pytest.raises(ConfigError, match="registered: core/"):
        get_preset("core/nope")


def test_e10_t10_presets_list_and_show_print_the_yaml() -> None:
    """`presets list` names every preset; `presets show` prints Appendix H verbatim."""
    out = runner.invoke(app, ["presets", "list"]).stdout
    assert "core/segmentation_gaussian_ltr" in out
    shown = runner.invoke(app, ["presets", "show", "core/segmentation_gaussian_ltr"])
    assert shown.exit_code == 0
    assert shown.stdout == get_preset("core/segmentation_gaussian_ltr").yaml
    assert "duration_dependent_jumps: false" in shown.stdout
    assert "bucket_edges: [256, 512, 1024, 2048]" in shown.stdout


def test_e10_t10_presets_show_unknown_exits_2() -> None:
    """An unknown preset is a `ConfigError` (exit 2)."""
    assert runner.invoke(app, ["presets", "show", "core/nope"]).exit_code == 2


@pytest.mark.parametrize("name", ["core/segmentation_gaussian_ltr", "core/two_regime_gaussian"])
def test_e10_t10_presets_write_yaml_and_fixture_validate(name: str, tmp_path: Path) -> None:
    """`presets write` emits YAML and the synthetic data it refers to; `validate-config` accepts it."""
    res = runner.invoke(app, ["presets", "write", name, str(tmp_path)])
    assert res.exit_code == 0, res.stdout
    yaml_path = tmp_path / f"{name.rpartition('/')[2]}.yaml"
    assert yaml_path.is_file()
    cfg = load_config(yaml_path)
    source_path = tmp_path / str(cfg.data.source.option("path"))
    assert source_path.is_file()
    frame = pd.read_parquet(source_path) if source_path.suffix == ".parquet" else pd.read_csv(source_path)
    for col in cfg.data.source.option("observation_columns"):
        assert col in frame.columns
    assert runner.invoke(app, ["validate-config", str(yaml_path)]).exit_code == 0


def test_e10_t10_segmentation_fixture_shape() -> None:
    """50 sessions of 150-400 steps, three observation columns, partially labelled with 4 regimes."""
    frame = get_preset("core/segmentation_gaussian_ltr").fixture(3)["data/sensor_windows.parquet"]  # type: ignore[misc]
    sizes = frame.groupby("session_id").size()
    assert len(sizes) == 50
    assert sizes.between(150, 400).all()
    labels = frame["activity"].dropna()
    assert 0.2 < len(labels) / len(frame) < 0.6
    assert set(labels.unique()) <= {0.0, 1.0, 2.0, 3.0}


def test_e10_t10_fixture_is_deterministic() -> None:
    """The same seed gives the same data."""
    fixture = get_preset("core/two_regime_gaussian").fixture
    assert fixture is not None
    pd.testing.assert_frame_equal(fixture(5)["data/two_regime.csv"], fixture(5)["data/two_regime.csv"])


def test_e10_t10_json_schema_validates_appendix_h_and_lists_registry_names() -> None:
    """The exported JSON Schema accepts Appendix H; registry-validated fields carry enums."""
    schema = json_schema()
    raw = json.loads(json.dumps(_SAFE_YAML.load(get_preset("core/segmentation_gaussian_ltr").yaml)))
    jsonschema.validate(raw, schema)
    defs = schema["$defs"]
    assert "gaussian" in defs["EmissionCfg"]["properties"]["family"]["enum"]
    per_state = defs["PerStateDurationCfg"]["properties"]["family"]["enum"]
    assert {"gamma", "geometric", "nonparametric"} <= set(per_state)
    assert None in defs["DurationCfg"]["properties"]["family"]["enum"]
    assert "boundary_f1" in defs["EvaluationSection"]["properties"]["metrics"]["items"]["enum"]
    assert "labels" in defs["EstimationSection"]["properties"]["init"]["items"]["enum"]


def test_e10_t10_config_schema_command_with_config_class(tmp_path: Path) -> None:
    """`config-schema --config-class` exports the schema of that class; a bad path exits 2."""
    out = tmp_path / "schema.json"
    ok = runner.invoke(
        app, ["config-schema", str(out), "--config-class", "hsmmkit.config.schema.ProjectConfig"]
    )
    assert ok.exit_code == 0, ok.stdout
    assert json.loads(out.read_text())["title"] == ProjectConfig.__name__
    bad = runner.invoke(app, ["config-schema", str(out), "--config-class", "hsmmkit.nope.Missing"])
    assert bad.exit_code == 2
    notcfg = runner.invoke(app, ["config-schema", str(out), "--config-class", "hsmmkit.presets.Preset"])
    assert notcfg.exit_code == 2


def test_e10_t10_regimekit_finance_config_schema_when_installed(tmp_path: Path) -> None:
    """With `regimekit` installed (downstream), its `FinanceConfig` exports and lists `gbm`."""
    pytest.importorskip("regimekit.config")
    out = tmp_path / "finance.json"
    res = runner.invoke(app, ["config-schema", str(out), "--config-class", "regimekit.config.FinanceConfig"])
    assert res.exit_code == 0
    assert "gbm" in json.loads(out.read_text())["$defs"]["FinanceEmissionCfg"]["properties"]["family"]["enum"]
