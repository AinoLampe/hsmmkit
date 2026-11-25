"""E10-T10: the final `hsmmkit` schema 1 (all Part II keys), validation rules and defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from hsmmkit.config.schema import ProjectConfig, parse_project_config
from hsmmkit.config.yaml_io import _SAFE_YAML
from hsmmkit.presets import get_preset
from hsmmkit.utils.errors import ConfigError
from tests.helpers.core_config import core_raw


def appendix_h() -> dict[str, Any]:
    """The raw dict of the Appendix H preset."""
    return dict(_SAFE_YAML.load(get_preset("core/segmentation_gaussian_ltr").yaml))


def _bad(raw: dict[str, Any], path: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse_project_config(raw)
    assert info.value.path is not None
    assert info.value.path.startswith(path), (info.value.path, str(info.value))


def test_e10_t10_appendix_h_validates() -> None:
    """The reference configuration of Part II Appendix H parses into a `ProjectConfig`."""
    cfg = parse_project_config(appendix_h())
    assert cfg.model.topology == "left_to_right"
    assert cfg.model.duration.per_state is not None
    assert len(cfg.model.duration.per_state) == 4
    assert cfg.data.labels is not None
    assert cfg.data.labels.column == "activity"
    assert cfg.online is not None
    assert cfg.online.alerts[0].p_off == 0.5
    assert cfg.evaluation.cv is not None
    assert cfg.estimation.method == "gem"
    assert cfg.model.d_max == 300


def test_e10_t10_part1_config_builds_unchanged_with_new_defaults(tmp_path: Path) -> None:
    """Every new key defaults to the Part I behaviour."""
    cfg = parse_project_config(core_raw(tmp_path))
    assert cfg.data.sequences is None
    assert cfg.data.labels is None
    assert cfg.model.topology == "full"
    assert cfg.model.left_censored is False
    assert cfg.model.initial_groups == "shared"
    assert cfg.model.emission.covariance == "diag"
    assert cfg.model.emission.mixture is None
    assert cfg.model.duration.per_state is None
    assert cfg.model.duration.duration_dependent_jumps is False
    assert cfg.estimation.method == "map"
    assert cfg.online is None
    assert cfg.evaluation.cv is None
    assert cfg.evaluation.grid is None
    assert cfg.evaluation.boundary_tolerance == 5


def test_e10_t10_backtest_and_forecaster_are_optional(tmp_path: Path) -> None:
    """Appendix H has no backtest and no covariate forecaster; the walk-forward commands ask for them."""
    raw = core_raw(tmp_path)
    del raw["evaluation"]["backtest"]
    del raw["forecast"]["covariate_forecaster"]
    cfg = parse_project_config(raw)
    assert cfg.forecast.covariate_forecaster.type == "freeze"
    with pytest.raises(ConfigError) as info:
        cfg.backtest_cfg()
    assert info.value.path == "evaluation.backtest"


def _mut(fn: Any) -> dict[str, Any]:  # noqa: ANN401
    raw = copy.deepcopy(appendix_h())
    fn(raw)
    return raw


RULES = {
    "bucket_edges_not_increasing": (
        lambda r: r["data"]["sequences"].update(bucket_edges=[256, 256]),
        "data.sequences.bucket_edges",
    ),
    "bucket_edges_below_one": (
        lambda r: r["data"]["sequences"].update(bucket_edges=[0, 5]),
        "data.sequences.bucket_edges",
    ),
    "labels_column_missing": (lambda r: r["data"]["labels"].update(column=None), "data.labels.column"),
    "labels_file_without_path": (
        lambda r: r["data"]["labels"].update(source="file", column=None),
        "data.labels.path",
    ),
    "per_state_needs_explicit": (lambda r: r["model"].update(engine="expanded"), "model.duration.per_state"),
    "per_state_length": (lambda r: r["model"]["duration"]["per_state"].pop(), "model.duration.per_state"),
    "family_and_per_state": (
        lambda r: r["model"]["duration"].update(family="geometric"),
        "model.duration.family",
    ),
    "ddj_needs_explicit": (
        lambda r: (
            r["model"].update(engine="expanded"),
            r["model"]["duration"].pop("per_state"),
            r["model"]["duration"].update(
                family="geometric", params={"rate": {"mode": "free"}}, duration_dependent_jumps=True
            ),
        ),
        "model.duration.duration_dependent_jumps",
    ),
    "banded_not_below_n": (lambda r: r["model"].update(topology={"banded": 4}), "model.topology.banded"),
    "matrix_shape": (
        lambda r: r["model"].update(topology={"matrix": [[0, 1], [1, 0]]}),
        "model.topology.matrix",
    ),
    "matrix_diagonal": (
        lambda r: r["model"].update(topology={"matrix": [[1, 1, 0, 0]] + [[0, 1, 0, 0]] * 3}),
        "model.topology.matrix",
    ),
    "p_off_not_below_p_on": (
        lambda r: r["online"]["alerts"][0].update(p_off=0.9),
        "online.alerts.p_off",
    ),
    "alert_regime_range": (lambda r: r["online"]["alerts"][0].update(regime=4), "online.alerts[0].regime"),
    "alert_two_kinds": (
        lambda r: r["online"]["alerts"][0].update(remaining_duration_below=5.0),
        "online.alerts",
    ),
    "labels_init_without_labels": (lambda r: r["data"].pop("labels"), "model.emission.params.mean.init"),
    "unknown_init": (lambda r: r["estimation"].update(init=["nope"]), "estimation.init[0]"),
    "unknown_per_state_family": (
        lambda r: r["model"]["duration"]["per_state"][0].update(family="nope"),
        "model.duration.per_state",
    ),
    "grid_n_states": (lambda r: r["evaluation"].update(grid={"n_states": [1]}), "evaluation.grid.n_states"),
    "cv_folds": (
        lambda r: r["evaluation"]["cv"].update(folds=1),
        "evaluation.cv.folds",
    ),
    "unknown_method": (lambda r: r["estimation"].update(method="sgd"), "estimation.method"),
}


@pytest.mark.parametrize("rule", sorted(RULES))
def test_e10_t10_validation_rule_names_the_path(rule: str) -> None:
    """Every Part II validation rule has a failing fixture whose `ConfigError.path` names the key."""
    mutate, path = RULES[rule]
    _bad(_mut(mutate), path)


def test_e10_t10_absorbing_topology_row_warns() -> None:
    """A matrix topology with an empty row is allowed (absorbing state) and warns, listing the states."""
    raw = appendix_h()
    raw["model"]["topology"] = {"matrix": [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]}
    cfg = parse_project_config(raw)
    assert any("absorbing" in w and "[3]" in w for w in cfg.warnings)


@pytest.mark.parametrize("method", ["vb", "mcmc"])
def test_e10_t10_non_conjugate_components_warn_for_vb_and_mcmc(method: str) -> None:
    """`method: vb|mcmc` warns on components without a conjugate prior, naming them."""
    raw = appendix_h()
    raw["estimation"]["method"] = method
    cfg = parse_project_config(raw)
    assert any("no conjugate prior" in w and "emission.gaussian" in w for w in cfg.warnings)


def test_e10_t10_core_config_class_is_project_config() -> None:
    """The preset parses with the default class."""
    assert isinstance(parse_project_config(appendix_h()), ProjectConfig)
