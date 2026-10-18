"""E1-T02b: `validate-config`/`config-schema` CLI, exit codes, and `--seed`/`--device` overrides."""

from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML
from typer.testing import CliRunner

from regimekit.cli.app import app

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "configs"
_YAML = YAML(typ="safe")
_runner = CliRunner()


# --- AC1: validate-config exits 0 on Appendix A, exits 2 with path on an invalid fixture ------


def test_e1_t02b_validate_config_appendix_a_exits_zero(tmp_path: Path) -> None:
    """`validate-config` on Appendix A exits 0 and prints a summary."""
    result = _runner.invoke(
        app, ["--out", str(tmp_path), "validate-config", str(FIXTURES / "appendix_a.yaml")]
    )
    assert result.exit_code == 0, result.output
    assert "n_states: 3" in result.output
    assert "engine: expanded" in result.output
    assert "emission family: gbm" in result.output
    assert "duration family: negbin" in result.output
    assert "warnings: (none)" in result.output


def test_e1_t02b_validate_config_invalid_fixture_exits_two_with_path(tmp_path: Path) -> None:
    """`validate-config` on the invalid `profile_core.yaml` fixture exits 2, path in output."""
    result = _runner.invoke(
        app,
        ["--out", str(tmp_path), "validate-config", str(FIXTURES / "invalid" / "profile_core.yaml")],
    )
    assert result.exit_code == 2
    assert "profile" in result.output


# --- AC5: --seed 7 appears as project.seed: 7 in the resolved dump ----------------------------


def test_e1_t02b_seed_override_appears_in_resolved_dump(tmp_path: Path) -> None:
    """`--seed 7` appears as `project.seed: 7` in the resolved dump."""
    result = _runner.invoke(
        app, ["--seed", "7", "--out", str(tmp_path), "validate-config", str(FIXTURES / "appendix_a.yaml")]
    )
    assert result.exit_code == 0, result.output
    data = _YAML.load(tmp_path / "config.resolved.yaml")
    assert data["project"]["seed"] == 7


def test_e1_t02b_device_override_appears_in_resolved_dump(tmp_path: Path) -> None:
    """`--device cpu` overrides `estimation.device` in the resolved dump."""
    result = _runner.invoke(
        app,
        ["--device", "cpu", "--out", str(tmp_path), "validate-config", str(FIXTURES / "appendix_a.yaml")],
    )
    assert result.exit_code == 0, result.output
    data = _YAML.load(tmp_path / "config.resolved.yaml")
    assert data["estimation"]["device"] == "cpu"


def test_e1_t02b_invalid_device_override_exits_two_with_config_error(tmp_path: Path) -> None:
    """`--device bogus` exits 2 with a clean `ConfigError` message, not a raw pydantic traceback."""
    result = _runner.invoke(
        app,
        ["--device", "bogus", "--out", str(tmp_path), "validate-config", str(FIXTURES / "appendix_a.yaml")],
    )
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "device" in result.output


def test_e1_t02b_log_level_override_appears_in_resolved_dump(tmp_path: Path) -> None:
    """`--log-level DEBUG` overrides `project.log_level` in the resolved dump."""
    result = _runner.invoke(
        app,
        [
            "--log-level",
            "DEBUG",
            "--out",
            str(tmp_path),
            "validate-config",
            str(FIXTURES / "appendix_a.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    data = _YAML.load(tmp_path / "config.resolved.yaml")
    assert data["project"]["log_level"] == "DEBUG"


def test_e1_t02b_out_option_controls_resolved_dump_directory(tmp_path: Path) -> None:
    """`--out DIR` writes `config.resolved.yaml` under `DIR`, not `project.output_dir`."""
    result = _runner.invoke(
        app, ["--out", str(tmp_path), "validate-config", str(FIXTURES / "appendix_a.yaml")]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.resolved.yaml").exists()


# --- config-schema command ---------------------------------------------------------------------


def test_e1_t02b_config_schema_command_writes_json(tmp_path: Path) -> None:
    """`config-schema OUT.json` writes `ProjectConfig`'s JSON-Schema."""
    out_path = tmp_path / "schema.json"
    result = _runner.invoke(app, ["config-schema", str(out_path)])
    assert result.exit_code == 0, result.output
    schema = json.loads(out_path.read_text())
    assert "profile" in schema["properties"]
