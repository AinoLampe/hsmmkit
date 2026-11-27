"""E13-T04 AC4: `decode(..., samples=R)` and `hsmmkit decode --samples R` write the documented artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from hsmmkit.cli.app import app
from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.batch import SequenceBatch
from hsmmkit.inference.decode import decode
from hsmmkit.project import Project
from hsmmkit.utils.errors import ConfigError
from tests.integration.test_e15_t04_stream import _write_world

_runner = CliRunner()


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A fitted two-regime Gaussian model and its data (shared with the stream tests' builder)."""
    return _write_world(tmp_path_factory.mktemp("decode_samples"), covariate=False)


def test_e13_t04_cli_decode_samples_writes_the_artifacts(world: dict[str, Any]) -> None:
    """`decode --samples 100`: `path_samples.parquet` (sample, t, regime) and `boundary_prob`."""
    out = world["root"] / "decoded"
    result = _runner.invoke(
        app,
        [
            "--out",
            str(out),
            "decode",
            str(world["config"]),
            "--model",
            str(world["model"]),
            "--samples",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    posterior = pq.read_table(out / "posterior.parquet").to_pandas()
    samples = pq.read_table(out / "path_samples.parquet").to_pandas()
    n_rows = posterior.shape[0]
    assert list(samples.columns) == ["sample", "t", "regime"]
    assert samples.shape[0] == 100 * n_rows
    assert samples["sample"].nunique() == 100
    assert set(samples["regime"].unique()) <= {0, 1}
    assert "boundary_prob" in posterior.columns
    prob = posterior["boundary_prob"].to_numpy()
    assert prob[0] == 0.0
    assert np.all((prob >= 0) & (prob <= 1))
    # the boundary probability is the regime-change frequency of the written samples
    paths = samples.pivot(index="sample", columns="t", values="regime").to_numpy()
    np.testing.assert_allclose(prob[1:], (paths[:, 1:] != paths[:, :-1]).mean(axis=0))
    # without --samples nothing extra is written
    plain = world["root"] / "decoded_plain"
    assert (
        _runner.invoke(
            app, ["--out", str(plain), "decode", str(world["config"]), "--model", str(world["model"])]
        ).exit_code
        == 0
    )
    assert not (plain / "path_samples.parquet").exists()
    assert "boundary_prob" not in pq.read_table(plain / "posterior.parquet").column_names


def test_e13_t04_python_api_is_deterministic_and_tracks_the_smoothed_marginals(world: dict[str, Any]) -> None:
    """`decode(..., samples=R, seed=s)` is deterministic; frequencies track the smoothed marginals."""
    project = Project(parse_project_config(world["raw"]), config_dir=world["root"], out_dir=world["out"])
    fit = Project.load(world["model"])
    data = project.load_data()
    a = decode(fit, data, samples=2000, seed=3)
    b = decode(fit, data, samples=2000, seed=3)
    c = decode(fit, data, samples=2000, seed=4)
    assert a.path_samples is not None
    assert b.path_samples is not None
    assert c.path_samples is not None
    np.testing.assert_array_equal(a.path_samples, b.path_samples)
    assert not np.array_equal(a.path_samples, c.path_samples)
    freq = (a.path_samples == 1).mean(axis=0)
    smoothed = np.asarray(a.smoothed)[:, 1]
    se = np.sqrt(np.maximum(smoothed * (1 - smoothed), 1e-12) / 2000)
    assert np.mean(np.abs(freq - smoothed) <= 3 * se + 1e-9) > 0.97  # 3 SE at (nearly) every step
    # batch decode gives per-sequence samples with the same shapes
    batch = SequenceBatch((data.window(0, 100), data.window(100, 140)), (128,))
    results = decode(fit, batch, samples=10, seed=1)
    assert [r.path_samples.shape for r in results if r.path_samples is not None] == [(10, 100), (10, 40)]
    with pytest.raises(ConfigError, match="samples"):
        decode(fit, data, samples=-1)
