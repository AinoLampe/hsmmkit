"""E12-T03 golden: `topology: full` reproduces the Part I model bitwise (absorbing regimes: unit tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hsmmkit.config.schema import parse_project_config
from hsmmkit.data.contracts import ObservationSet
from hsmmkit.data.prepare import Preparer
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.model.build import build_model_spec
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.model.transition import JumpTransition
from hsmmkit.statespace.builders import build_expanded
from tests.helpers.core_config import core_raw


def _model(tmp_path: Path, **model: Any) -> tuple[Any, Any, Any]:  # noqa: ANN401
    raw = core_raw(tmp_path)
    raw["model"]["n_states"] = 3
    raw["model"]["emission"]["params"] = {
        "mean": {"mode": "free", "init": [0.0, 0.1, 0.2]},
        "scale": {"mode": "free", "init": [0.5, 0.8, 1.2]},
    }
    raw["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [12.0, 12.0, 12.0]
    raw["model"].update(model)
    cfg = parse_project_config(raw)
    y = np.random.default_rng(3).normal(size=(120, 1)) * 0.6
    times = pd.bdate_range("2020-01-01", periods=120).to_numpy().astype("datetime64[ns]")
    obs = ObservationSet(("x",), times, y, "memory", "s")
    data = Preparer(cfg.data).fit(obs, [], train_end=times[-1]).transform(obs, [])
    return cfg, data, build_model_spec(cfg, data)


def test_e12_t03_full_topology_is_the_part_i_model_bitwise(tmp_path: Path) -> None:
    """AC4: `topology: full` (default and explicit) builds the Part I specs; bitwise equal log-likelihood."""
    _, data, default = _model(tmp_path)
    _, _, explicit = _model(tmp_path, topology="full")
    key = [(sp.name, sp.shape, sp.domain, sp.mode) for sp in default.param_specs]
    assert key == [(sp.name, sp.shape, sp.domain, sp.mode) for sp in explicit.param_specs]
    for sa, sb in zip(default.param_specs, explicit.param_specs, strict=True):
        assert np.array_equal(sa.init, sb.init)
    assert getattr(default.transition, "topology", None) is None
    assert isinstance(default.transition, JumpTransition)
    assert default.transition == explicit.transition
    names = [s.name for s in default.param_specs if s.name.startswith("transition.")]
    assert names == ["transition.pi_rows"]  # the Part I layout, not the flat topology parameters
    mgr = ParamManager(default.param_specs)
    theta = mgr.unflatten(mgr.init_u())
    ll_a = ExpandedHMMEngine().loglik(theta, default, build_expanded(default), data)
    ll_b = ExpandedHMMEngine().loglik(theta, explicit, build_expanded(explicit), data)
    assert float(ll_a) == float(ll_b)


def test_e12_t03_banded_covering_every_jump_equals_the_full_model(tmp_path: Path) -> None:
    """A matrix topology that allows every jump has the full model's likelihood at equivalent parameters."""
    _, data, full = _model(tmp_path)
    _, _, matrix = _model(tmp_path, topology={"matrix": [[0, 1, 1], [1, 0, 1], [1, 1, 0]]})
    assert matrix.transition.topology is None  # recognised as full: Part I layout
    mgr = ParamManager(full.param_specs)
    theta = mgr.unflatten(mgr.init_u())
    a = ExpandedHMMEngine().loglik(theta, full, build_expanded(full), data)
    b = ExpandedHMMEngine().loglik(theta, matrix, build_expanded(matrix), data)
    assert float(a) == float(b)
