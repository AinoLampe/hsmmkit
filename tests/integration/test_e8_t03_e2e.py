"""E8-T03: the six-command flow on the example config, replayed offline on synthetic look-alike data.

CI uses synthetic data (`examples/synthetic.py`): provider data is never committed (NFR-08). Real-data runs
are `tests/live/test_e8_t03_real_data.py` (`-m live`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from examples.synthetic import write_offline_project
from regimekit.config.yaml_io import load_config
from regimekit.finance import FinanceConfig
from regimekit.finance.cli import app
from regimekit.persistence.store import FILES

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

_runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "configs"
DAX = EXAMPLES / "dax_3state_gbm_nb.yaml"
SPX = EXAMPLES / "spx_3state_gbm_nb.yaml"


def _cli(*args: str) -> Result:
    return _runner.invoke(app, list(args))


@pytest.fixture(scope="module")
def flow(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """`(config, results dir, model bundle)` after validate-config, fetch, fit, decode, forecast, backtest."""
    work = tmp_path_factory.mktemp("e8t03")
    config = write_offline_project(work, DAX)
    out = work / "runs"
    name = load_config(config, FinanceConfig).project.name
    model = out / f"{name}.rkmodel"
    steps = [
        ("validate-config", str(config)),
        ("fetch", str(config)),
        ("fit", str(config)),
        ("decode", str(config), "--model", str(model)),
        ("forecast", str(config), "--model", str(model)),
        ("backtest", str(config)),
    ]
    for step in steps:
        result = (
            _cli("--offline", "--out", str(out), *step)
            if step[0] != "fetch"
            else _cli("--out", str(out), *step)
        )
        assert result.exit_code == 0, f"{step[0]}: {result.output}"
    return config, out, model


def test_e8_t03_example_configs_are_valid_and_differ_in_the_index() -> None:
    """Both shipped example configs load; they are Appendix A with `^GDAXI` and `^GSPC`."""
    dax, spx = load_config(DAX, FinanceConfig), load_config(SPX, FinanceConfig)
    assert dax.data.primary.symbols == ["^GDAXI"]
    assert spx.data.primary.symbols == ["^GSPC"]
    assert dax.model == spx.model
    assert dax.project.name != spx.project.name


def test_e8_t03_offline_flow_writes_every_artifact(flow: tuple[Path, Path, Path]) -> None:
    """AC1: bundle files plus posterior, forecast, backtest metrics, strategy, report and quality."""
    _, out, model = flow
    for name in FILES:
        assert (model / name).is_file(), f"bundle file {name}"
    for name in (
        "posterior.parquet",
        "forecast.parquet",
        "backtest_metrics.parquet",
        "strategy.parquet",
        "report.md",
        "quality.json",
    ):
        assert (out / name).is_file(), name


def test_e8_t03_two_fits_give_bitwise_equal_params(flow: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """AC2: a second `fit` of the same config and seed (42) writes byte-identical `params.safetensors`."""
    config, _, model = flow
    assert load_config(config, FinanceConfig).project.seed == 42
    again = tmp_path / "again"
    result = _cli("--out", str(again), "fit", str(config))
    assert result.exit_code == 0, result.output
    second = again / model.name / "params.safetensors"
    assert second.read_bytes() == (model / "params.safetensors").read_bytes()
