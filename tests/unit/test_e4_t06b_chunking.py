"""E4-T06b: chunked forward-backward, memory bound, irregular-grid warning, `estimation.chunk_size`."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

from regimekit.config.schema import EstimationSection, ProjectConfig
from regimekit.config.yaml_io import dump_resolved, export_json_schema, load_config
from regimekit.engine import explicit_duration
from regimekit.engine.explicit_duration import ExplicitDurationEngine, _forward_backward_core
from regimekit.utils.errors import ConfigError
from tests.helpers.engine_fixtures import make_explicit_scenario

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "configs"


@pytest.mark.parametrize("chunk", [64, 1, 999, 1000, 5000])
def test_e4_t06b_chunked_equals_unchunked_t1000(chunk: int) -> None:
    """AC3: chunked forward-backward equals the full one to 1e-10 (T=1000, chunk 64 and edge sizes)."""
    sc = make_explicit_scenario(3, 20, 1000, family="gamma", seed=2, mean_steps=8.0)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    full = ExplicitDurationEngine().forward_backward(*args)
    part = ExplicitDurationEngine(chunk_size=chunk).forward_backward(*args)

    assert abs(float(full.loglik) - float(part.loglik)) <= 1e-10 * abs(float(full.loglik))
    np.testing.assert_allclose(
        np.asarray(part.regime_smoothed()), np.asarray(full.regime_smoothed()), rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(part.regime_filtered()), np.asarray(full.regime_filtered()), rtol=0, atol=1e-10
    )
    assert np.asarray(part.smoothed).shape == (1000, 3)
    assert part.index.n_expanded == 3
    assert part.engine == "explicit"


def test_e4_t06b_chunked_handles_entry_time_covariates() -> None:
    """Chunking with `(T, N, D_max)` pmf and `(T, N, N)` jump tables still matches the full pass."""
    sc = make_explicit_scenario(3, 8, 90, family="weibull", n_cov_d=1, n_cov_pi=1, seed=9)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    full = ExplicitDurationEngine().forward_backward(*args)
    part = ExplicitDurationEngine(chunk_size=16).forward_backward(*args)
    np.testing.assert_allclose(
        np.asarray(part.regime_smoothed()), np.asarray(full.regime_smoothed()), rtol=0, atol=1e-10
    )


def test_e4_t06b_auto_chunks_above_the_cell_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """`chunk_size=None` switches to chunks once `T * N * D_max` exceeds the limit (no `NumericalError`)."""
    sc = make_explicit_scenario(3, 10, 120, family="negbin", seed=4)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    full = ExplicitDurationEngine().forward_backward(*args)
    monkeypatch.setattr(explicit_duration, "MAX_CELLS", 100.0)
    auto = ExplicitDurationEngine().forward_backward(*args)
    assert np.asarray(auto.smoothed).shape == (120, 3)
    np.testing.assert_allclose(
        np.asarray(auto.regime_smoothed()), np.asarray(full.regime_smoothed()), rtol=0, atol=1e-10
    )
    # `loglik` and `viterbi` never need full tables, so they run above the limit as well
    assert float(ExplicitDurationEngine().loglik(*args)) == pytest.approx(float(full.loglik), abs=1e-10)
    assert np.asarray(ExplicitDurationEngine().viterbi(*args)).shape == (120,)


def test_e4_t06b_invalid_chunk_size_is_rejected() -> None:
    """`chunk_size < 1` raises `ConfigError` (path `estimation.chunk_size`)."""
    sc = make_explicit_scenario(2, 4, 20, family="geometric", seed=1)
    with pytest.raises(ConfigError) as exc:
        ExplicitDurationEngine(chunk_size=0).forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    assert exc.value.path == "estimation.chunk_size"


def test_e4_t06b_full_forward_backward_above_limit_without_chunking_is_impossible() -> None:
    """`forward` (full `log_alpha`) still refuses tables above the limit."""
    sc = make_explicit_scenario(2, 4, 30, family="geometric", seed=1)
    explicit_duration.MAX_CELLS, saved = 10.0, explicit_duration.MAX_CELLS
    try:
        with pytest.raises(explicit_duration.NumericalError, match="chunked scan"):
            ExplicitDurationEngine().forward(sc.theta, sc.spec, sc.space, sc.data)
    finally:
        explicit_duration.MAX_CELLS = saved


def test_e4_t06b_chunked_pass_at_6e7_cells_stays_below_the_full_table_size() -> None:
    """AC3: T*N*D_max = 6e7 runs chunked; the compiled temp size stays far below one `(T, N, D_max)` array."""
    t_len, n, d_max = 60_000, 4, 250
    sc = make_explicit_scenario(n, d_max, t_len, family="negbin", seed=0, mean_steps=60.0)
    args = (sc.theta, sc.spec, sc.space, sc.data)
    full_bytes = t_len * n * d_max * 8
    assert t_len * n * d_max == 6e7

    compiled = _forward_backward_core.lower(*args, 1024).compile()
    analysis = compiled.memory_analysis()
    assert analysis is not None
    assert analysis.temp_size_in_bytes < full_bytes / 4

    post = ExplicitDurationEngine().forward_backward(*args)  # auto-chunked
    assert np.isfinite(float(post.loglik))
    np.testing.assert_allclose(np.asarray(post.regime_smoothed()).sum(axis=1), 1.0, rtol=0, atol=1e-9)


def test_e4_t06b_irregular_grid_warns_exactly_once_per_call() -> None:
    """AC4: `viterbi`, chunked and full `forward_backward` each warn once on an irregular grid."""
    sc = make_explicit_scenario(3, 6, 40, family="gamma", seed=3, irregular=True)
    calls = (
        ExplicitDurationEngine().viterbi,
        ExplicitDurationEngine(chunk_size=8).forward_backward,
        ExplicitDurationEngine().forward_backward,
    )
    for call in calls:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(sc.theta, sc.spec, sc.space, sc.data)
        assert sum("regular grid" in str(w.message) for w in caught) == 1


# --- config field `estimation.chunk_size` (CLAUDE.md §3.4) ------------------------------------

_YAML = YAML(typ="safe")


def test_e4_t06b_chunk_size_config_default_and_validation() -> None:
    """Default `None` keeps Appendix A valid; positive integers only."""
    assert EstimationSection().chunk_size is None
    assert EstimationSection(chunk_size=64).chunk_size == 64
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        EstimationSection(chunk_size=0)
    assert isinstance(load_config(FIXTURES / "appendix_a.yaml"), ProjectConfig)


def test_e4_t06b_chunk_size_in_json_schema_and_resolved_dump(tmp_path: Path) -> None:
    """The JSON Schema lists `chunk_size` and the resolved dump writes the defaulted value."""
    schema_path = tmp_path / "schema.json"
    export_json_schema(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    props = schema["$defs"]["EstimationSection"]["properties"]
    assert "chunk_size" in props
    assert {"type": "null"} in props["chunk_size"]["anyOf"]

    resolved = tmp_path / "config.resolved.yaml"
    dump_resolved(load_config(FIXTURES / "appendix_a.yaml"), resolved)
    assert "chunk_size" in _YAML.load(resolved)["estimation"]
