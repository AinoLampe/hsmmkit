"""E10-T02b: the finance migration 1 -> 2, rule by rule (FR-G07)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from ruamel.yaml import YAML

from regimekit.config.migrate import migrate
from regimekit.finance import FinanceConfig
from regimekit.finance.config import migrate_1_to_2
from regimekit.utils.errors import ConfigError

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "configs"


@pytest.fixture
def v1() -> dict[str, Any]:
    """A fresh copy of the Part I Appendix A configuration."""
    return copy.deepcopy(YAML(typ="safe").load(FIXTURES / "appendix_a.yaml"))


def test_e10_t02b_migration_drops_profile_and_sets_the_version(v1: dict[str, Any]) -> None:
    """Migration drops profile and sets the version."""
    out = migrate_1_to_2(v1)
    assert "profile" not in out
    assert out["schema_version"] == 2


def test_e10_t02b_migration_rejects_profile_core(v1: dict[str, Any]) -> None:
    """Migration rejects profile core."""
    v1["profile"] = "core"
    with pytest.raises(ConfigError) as err:
        migrate_1_to_2(v1)
    assert err.value.path == "profile"


def test_e10_t02b_migration_moves_primary_to_source(v1: dict[str, Any]) -> None:
    """Migration moves primary to source."""
    primary = dict(v1["data"]["primary"])
    out = migrate_1_to_2(v1)
    assert "primary" not in out["data"]
    assert out["data"]["source"] == {"kind": "provider", **primary}


def test_e10_t02b_migration_wraps_covariate_provider_and_series(v1: dict[str, Any]) -> None:
    """Migration wraps covariate provider and series."""
    first = dict(v1["data"]["covariates"][0])
    out = migrate_1_to_2(v1)
    cov = out["data"]["covariates"][0]
    assert "provider" not in cov
    assert "series" not in cov
    assert cov["source"] == {"kind": "provider", "provider": first["provider"], "series": first["series"]}


def test_e10_t02b_migration_moves_bars_per_day_to_clock_options(v1: dict[str, Any]) -> None:
    """Migration moves bars per day to clock options."""
    v1["data"]["bars_per_day"] = 3
    out = migrate_1_to_2(v1)
    assert "bars_per_day" not in out["data"]
    assert out["data"]["clock_options"] == {"bars_per_day": 3}


def test_e10_t02b_migration_renames_preparation_keys(v1: dict[str, Any]) -> None:
    """Migration renames preparation keys."""
    v1["data"]["preparation"] = {"returns": "log", "missing_returns": "error", "standardize_covariates": True}
    prep = migrate_1_to_2(v1)["data"]["preparation"]
    assert prep == {
        "observation_transform": "log_returns",
        "missing_observations": "error",
        "standardize_covariates": True,
    }


def test_e10_t02b_migration_renames_backtest_windows(v1: dict[str, Any]) -> None:
    """Migration renames backtest windows."""
    bt = v1["evaluation"]["backtest"]
    bt["rolling_years"] = 4
    years = bt["initial_train_years"]
    out = migrate_1_to_2(v1)["evaluation"]["backtest"]
    assert out["initial_train"] == years
    assert out["rolling_window"] == 4
    assert "initial_train_years" not in out
    assert "rolling_years" not in out


def test_e10_t02b_migration_declares_scale_by_dt_for_student_t(v1: dict[str, Any]) -> None:
    """Migration declares scale by dt for student t."""
    v1["model"]["emission"]["family"] = "student_t"
    assert migrate_1_to_2(v1)["model"]["emission"]["scale_by_dt"] is True
    v1["model"]["emission"]["scale_by_dt"] = False
    assert migrate_1_to_2(v1)["model"]["emission"]["scale_by_dt"] is False


def test_e10_t02b_migration_leaves_gbm_alone(v1: dict[str, Any]) -> None:
    """Migration leaves gbm alone."""
    assert "scale_by_dt" not in migrate_1_to_2(v1)["model"]["emission"]


def test_e10_t02b_migrate_applies_the_registered_step_for_finance(v1: dict[str, Any]) -> None:
    """Migrate applies the registered step for finance."""
    assert migrate(v1, FinanceConfig)["schema_version"] == 2
    with pytest.raises(ConfigError, match="newer than library"):
        migrate({"schema_version": 3}, FinanceConfig)


def test_e10_t02b_migrated_v1_files_validate(v1: dict[str, Any]) -> None:
    """Migrated v1 files validate."""
    from regimekit.config.schema import parse_project_config

    cfg = parse_project_config(migrate(v1, FinanceConfig), FinanceConfig)
    assert cfg.data.source.option("symbols") == ["^GDAXI"]
    assert cfg.data.covariates[0].provider == "fred"
    assert cfg.data.covariates[0].series == "VIXCLS"
