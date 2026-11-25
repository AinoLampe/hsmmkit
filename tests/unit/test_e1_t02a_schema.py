"""E1-T02a: config schema parsing, cross-field validation rules, round-trip."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

from regimekit.utils.errors import ConfigError
from tests.helpers.finance_config import parse_v1 as parse_project_config

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "configs"
_YAML = YAML(typ="safe")


def _load(name: str) -> dict[str, Any]:
    return _YAML.load(FIXTURES / name)  # type: ignore[no-any-return]


@pytest.fixture
def appendix_a() -> dict[str, Any]:
    """A deep copy of Appendix A's reference config, safe for tests to mutate."""
    return copy.deepcopy(_load("appendix_a.yaml"))


# --- AC1: Appendix A parses without error, no warnings, correct defaults --------------------


def test_e1_t02a_appendix_a_parses_without_error(appendix_a: dict[str, Any]) -> None:
    """Appendix A parses into `ProjectConfig` without error and yields no warnings."""
    config = parse_project_config(appendix_a)
    assert config.warnings == []


def test_e1_t02a_appendix_a_profile_and_clock_defaults(appendix_a: dict[str, Any]) -> None:
    """The migrated model has schema version 2 and `data.clock == 'trading'`."""
    config = parse_project_config(appendix_a)
    assert config.schema_version == 2
    assert config.data.clock == "trading"


# --- AC5: profile='core' fails; steps_clock fixture validates; days-under-steps fails; -------
# --- bars_per_day=2 under steps fails ---------------------------------------------------------


def test_e1_t02a_profile_core_fails_on_profile_path(appendix_a: dict[str, Any]) -> None:
    """`profile: core` fails with path `profile`."""
    appendix_a["profile"] = "core"
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "profile"


def test_e1_t02a_profile_core_invalid_fixture_fails_on_profile_path() -> None:
    """The standalone `invalid/profile_core.yaml` fixture also fails with path `profile`."""
    data = _load("invalid/profile_core.yaml")
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(data)
    assert exc_info.value.path == "profile"


def test_e1_t02a_steps_clock_fixture_validates() -> None:
    """`steps_clock.yaml` (Appendix A with `clock: steps`, `init_mean_sojourn_steps`) validates."""
    config = parse_project_config(_load("steps_clock.yaml"))
    assert config.data.clock == "steps"
    assert config.warnings == []


def test_e1_t02a_steps_clock_with_days_fails_on_param_cfg_path() -> None:
    """The steps-clock fixture with `_days` instead of `_steps` fails with the `ParamCfg` path."""
    data = copy.deepcopy(_load("steps_clock.yaml"))
    data["model"]["duration"]["params"]["rate"] = {
        "mode": "free",
        "init_mean_sojourn_days": [60, 120, 30],
        "prior": {"type": "lognormal", "sd": 1.0},
    }
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(data)
    assert exc_info.value.path == "model.duration.params.rate.init_mean_sojourn_days"


def test_e1_t02a_bars_per_day_2_under_steps_fails() -> None:
    """`bars_per_day: 2` under `clock: steps` fails with path `data.clock_options.bars_per_day`."""
    data = copy.deepcopy(_load("steps_clock.yaml"))
    data["data"]["bars_per_day"] = 2
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(data)
    assert exc_info.value.path == "data.clock_options.bars_per_day"


# --- AC3: every validation rule has a failing fixture + ConfigError.path test ----------------


def test_e1_t02a_per_state_length_mismatch_r(appendix_a: dict[str, Any]) -> None:
    """`duration.r` with the wrong length fails on path `model.duration.r`."""
    appendix_a["model"]["duration"]["r"] = [3, 3]
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.duration.r"


def test_e1_t02a_per_state_length_mismatch_param_values(appendix_a: dict[str, Any]) -> None:
    """A `ParamCfg` list field with the wrong length fails on its own dotted path."""
    appendix_a["model"]["emission"]["params"]["mu"]["values"] = [-0.3, 0.0]
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.emission.params.mu.values"


def test_e1_t02a_mutually_exclusive_sojourn_fields(appendix_a: dict[str, Any]) -> None:
    """At most one of `init_mean_sojourn_days`/`init_mean_sojourn_steps` per `ParamCfg`."""
    appendix_a["model"]["duration"]["params"]["rate"]["init_mean_sojourn_steps"] = [60, 120, 30]
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.duration.params.rate.init_mean_sojourn_days"


def test_e1_t02a_covariate_not_in_data_covariates_fails(appendix_a: dict[str, Any]) -> None:
    """A referenced covariate that isn't declared in `data.covariates` fails."""
    appendix_a["model"]["transition"]["covariates"] = ["nonexistent"]
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.transition.covariates"


def test_e1_t02a_covariate_missing_use_in_entry_point_fails(appendix_a: dict[str, Any]) -> None:
    """A covariate that doesn't list the referencing entry point in its own `use_in` fails."""
    appendix_a["data"]["covariates"][1]["use_in"] = ["duration"]  # term_spread, used by transition
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.transition.covariates"


def test_e1_t02a_emission_drift_covariate_missing_use_in_fails(appendix_a: dict[str, Any]) -> None:
    """An `emission.covariates.drift` reference not listing `emission` in its `use_in` fails."""
    appendix_a["model"]["emission"]["covariates"]["drift"] = ["log_vix"]  # use_in: [transition, duration]
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.emission.covariates.drift"


def test_e1_t02a_emission_log_vol_covariate_missing_use_in_fails(appendix_a: dict[str, Any]) -> None:
    """An `emission.covariates.log_vol` reference not listing `emission` in its `use_in` fails."""
    appendix_a["model"]["emission"]["covariates"]["log_vol"] = ["term_spread"]  # use_in: [transition]
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.emission.covariates.log_vol"


def test_e1_t02a_generator_mode_with_geometric_but_r_not_all_one_fails(appendix_a: dict[str, Any]) -> None:
    """`mode='generator'` with `family='geometric'` but `r` not all 1 still fails."""
    appendix_a["model"]["transition"]["mode"] = "generator"
    appendix_a["model"]["duration"] = {
        "family": "geometric",
        "r": [1, 2, 1],
        "params": {"a": {"mode": "free", "init": [0.1, 0.2, 0.3]}},
        "covariates": [],
    }
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.transition.mode"


def test_e1_t02a_d_max_defaults_to_cap_without_any_sojourn_info(appendix_a: dict[str, Any]) -> None:
    """`d_max` defaults to the hard cap when no `ParamCfg` gives sojourn/init information."""
    del appendix_a["model"]["d_max"]
    appendix_a["model"]["duration"]["params"]["rate"] = {"mode": "fixed", "values": [0.1, 0.1, 0.1]}
    config = parse_project_config(appendix_a)
    assert config.model.d_max == 500


def test_e1_t02a_expected_sojourn_from_plain_init_under_steps_clock() -> None:
    """Under `clock='steps'`, a plain `init` (no `init_mean_sojourn_*`) sizes `d_max` directly."""
    data = copy.deepcopy(_load("steps_clock.yaml"))
    del data["model"]["d_max"]
    data["model"]["duration"]["params"]["rate"] = {"mode": "free", "init": [10.0, 10.0, 10.0]}
    config = parse_project_config(data)
    assert config.model.d_max == 100


def test_e1_t02a_lag_less_than_one_fails(appendix_a: dict[str, Any]) -> None:
    """`lag < 1` fails."""
    appendix_a["data"]["covariates"][0]["lag"] = 0
    with pytest.raises(ConfigError):
        parse_project_config(appendix_a)


def test_e1_t02a_order_greater_than_two_warns(appendix_a: dict[str, Any]) -> None:
    """`transition.order > 2` produces a warning, not an error."""
    appendix_a["model"]["transition"]["order"] = 3
    config = parse_project_config(appendix_a)
    assert any("order" in w for w in config.warnings)


def test_e1_t02a_order_less_than_one_fails(appendix_a: dict[str, Any]) -> None:
    """`transition.order < 1` fails."""
    appendix_a["model"]["transition"]["order"] = 0
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.transition.order"


def test_e1_t02a_engine_explicit_negbin_covariates_warns_and_sets_flag(appendix_a: dict[str, Any]) -> None:
    """`engine=explicit` + `duration.family=negbin` + duration covariates warns and sets the flag."""
    appendix_a["model"]["engine"] = "explicit"
    config = parse_project_config(appendix_a)
    assert config.model.duration.duration_covariates_at_entry is True
    assert any("entry-time evaluation" in w for w in config.warnings)


def test_e1_t02a_d_max_default_from_expected_sojourn(appendix_a: dict[str, Any]) -> None:
    """`d_max` defaults to `min(500, 10 * expected_sojourn_steps)`."""
    del appendix_a["model"]["d_max"]
    appendix_a["model"]["duration"]["params"]["rate"]["init_mean_sojourn_days"] = [10, 10, 10]
    config = parse_project_config(appendix_a)
    assert config.model.d_max == 100


def test_e1_t02a_d_max_below_warn_threshold_warns(appendix_a: dict[str, Any]) -> None:
    """`d_max < 2 * max(expected_sojourn_steps)` produces a warning."""
    appendix_a["model"]["d_max"] = 10
    config = parse_project_config(appendix_a)
    assert any("d_max" in w for w in config.warnings)


def test_e1_t02a_generator_mode_requires_geometric_and_r_all_one(appendix_a: dict[str, Any]) -> None:
    """`mode='generator'` is only valid with `family='geometric'` and `r` all 1."""
    appendix_a["model"]["transition"]["mode"] = "generator"
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "model.transition.mode"


def test_e1_t02a_generator_mode_with_geometric_r_all_one_validates(appendix_a: dict[str, Any]) -> None:
    """`mode='generator'` with `family='geometric'` and `r` all 1 validates."""
    appendix_a["model"]["transition"]["mode"] = "generator"
    appendix_a["model"]["duration"] = {
        "family": "geometric",
        "r": [1, 1, 1],
        "params": {"a": {"mode": "free", "init": [0.1, 0.2, 0.3]}},
        "covariates": [],
    }
    config = parse_project_config(appendix_a)
    assert config.model.transition.mode == "generator"


@pytest.mark.parametrize(
    ("mode", "field_to_clear"),
    [("fixed", "values"), ("bounded", "bounds"), ("tied", "tie"), ("prior", "prior")],
)
def test_e1_t02a_mode_requires_its_field(appendix_a: dict[str, Any], mode: str, field_to_clear: str) -> None:
    """`values` required for `fixed`, `bounds` for `bounded`, `tie` for `tied`, `prior` for `prior`."""
    appendix_a["model"]["emission"]["params"]["mu"] = {"mode": mode}
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == f"model.emission.params.mu.{field_to_clear}"


def test_e1_t02a_quantiles_must_be_strictly_increasing(appendix_a: dict[str, Any]) -> None:
    """`quantiles` out of order fails on path `quantiles`."""
    appendix_a["forecast"]["quantiles"] = [0.5, 0.1, 0.9]
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "quantiles"


def test_e1_t02a_quantiles_must_be_in_open_unit_interval(appendix_a: dict[str, Any]) -> None:
    """`quantiles` outside `(0, 1)` fails on path `quantiles`."""
    appendix_a["forecast"]["quantiles"] = [0.0, 0.5, 0.9]
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "quantiles"


# --- AC4: unknown key at any nesting level fails with its path -------------------------------


def test_e1_t02a_unknown_top_level_key_fails(appendix_a: dict[str, Any]) -> None:
    """An unknown top-level key fails with its path."""
    appendix_a["bogus_top_level"] = 1
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "bogus_top_level"


def test_e1_t02a_unknown_nested_key_fails_with_dotted_path(appendix_a: dict[str, Any]) -> None:
    """An unknown deeply-nested key fails with its dotted path."""
    appendix_a["project"]["bogus_key"] = 1
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "project.bogus_key"


def test_e1_t02a_unknown_key_inside_list_item_fails(appendix_a: dict[str, Any]) -> None:
    """An unknown key inside a list item (e.g. a covariate entry) fails with its path."""
    appendix_a["data"]["covariates"][0]["bogus"] = 1
    with pytest.raises(ConfigError) as exc_info:
        parse_project_config(appendix_a)
    assert exc_info.value.path == "data.covariates.0.bogus"


# --- AC2: model_dump() round-trips through model_validate unchanged -------------------------


def test_e1_t02a_model_dump_round_trips(appendix_a: dict[str, Any]) -> None:
    """`ProjectConfig.model_dump()` round-trips through `model_validate` unchanged."""
    config = parse_project_config(appendix_a)
    dumped = config.model_dump(mode="json")
    reloaded = type(config).model_validate(dumped)
    assert reloaded == config
