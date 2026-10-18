"""E4-T07: GPU tests skip (not fail) without a GPU; `estimation.parallel_scan` config field."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest
from ruamel.yaml import YAML

from regimekit.config.schema import EstimationSection
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "configs"
_YAML = YAML(typ="safe")


def _has_gpu() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


@pytest.mark.skipif(_has_gpu(), reason="a GPU is present: the GPU tests run instead of skipping")
def test_e4_t07_gpu_tests_are_skipped_not_failed_on_cpu_only_hosts() -> None:
    """AC2: `pytest -m gpu` on a CPU-only host exits 0 with every GPU test skipped."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "gpu", "tests/golden/test_e4_t07_gpu.py", "-q", "--no-cov"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 skipped" in result.stdout
    assert "failed" not in result.stdout


def test_e4_t07_gpu_tests_are_excluded_from_the_default_run() -> None:
    """The default `addopts` deselect `gpu` and `benchmark` tests."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "not benchmark and not gpu" in text


def test_e4_t07_parallel_scan_config_default_and_schema(tmp_path: Path) -> None:
    """`estimation.parallel_scan` defaults to `False`, is in the JSON Schema and in the resolved dump."""
    assert EstimationSection().parallel_scan is False
    assert EstimationSection(parallel_scan=True).parallel_scan is True

    schema_path = tmp_path / "schema.json"
    export_json_schema(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    prop = schema["$defs"]["EstimationSection"]["properties"]["parallel_scan"]
    assert prop["type"] == "boolean"
    assert prop["default"] is False

    resolved = tmp_path / "config.resolved.yaml"
    dump_resolved(load_config(FIXTURES / "appendix_a.yaml"), resolved)
    assert _YAML.load(resolved)["estimation"]["parallel_scan"] is False
