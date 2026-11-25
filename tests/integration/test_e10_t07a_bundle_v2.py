"""E10-T07a: bundle v2 (`provenance.json`, `extensions/`), v1 read compatibility, missing extensions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.data.prepare import Preparer
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.estimation.objective import canonical_data
from regimekit.finance import FinanceConfig, extension_payloads
from regimekit.model.build import build_model_spec
from regimekit.persistence.store import BUNDLE_VERSION, FILES, ModelStore
from regimekit.statespace.builders import build_expanded
from regimekit.utils.errors import PersistenceError
from tests.helpers import backtest_fixtures as bf

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bundles"


def _expected() -> dict[str, object]:
    return json.loads((FIXTURE / "v0_2_0.expected.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_e10_t07a_the_v0_2_0_bundle_loads_through_the_v1_path_and_reproduces_the_loglik(
    tmp_path: Path,
) -> None:
    """AC3: the bundle written by tag v0.2.0 (`bundle_version 1`) loads and gives its stored loglik."""
    manifest = json.loads((FIXTURE / "v0_2_0.rkmodel" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_version"] == 1
    assert manifest["profile"] == "finance"
    fit = ModelStore.load(FIXTURE / "v0_2_0.rkmodel", FinanceConfig)
    expected = _expected()
    assert fit.loglik == expected["loglik"]
    assert np.asarray(fit.uparams).tolist() == expected["uparams"]
    assert list(fit.restarts.columns)[:2] == ["restart", "init_method"]  # the v1 column is renamed on load
    assert set(fit.params) >= {"emission.mu", "emission.sigma"}  # the Part I parameter names load

    # the loaded parameters give the stored log-likelihood on the data of the bundle
    returns = bf.simulate_returns(199, seed=5)
    obs = bf.observations(returns)
    cfg = bf.make_config(tmp_path, with_covariate=False, restarts=1)
    data = Preparer(cfg.data).fit(obs, [], train_end=obs.times[-1]).transform(obs, [])
    spec = build_model_spec(cfg, data)
    space = build_expanded(spec)
    loglik = float(ExpandedHMMEngine().loglik(fit.params, spec, space, canonical_data(data)))
    assert loglik == pytest.approx(float(expected["loglik"]), rel=1e-10)  # bitwise on the generating machine


def test_e10_t07a_saving_the_v1_bundle_again_writes_v2_with_the_finance_extension(tmp_path: Path) -> None:
    """AC3: re-saving writes `bundle_version 2`, `provenance.json` and `extensions/regimekit.json`."""
    fit = ModelStore.load(FIXTURE / "v0_2_0.rkmodel", FinanceConfig)
    returns = bf.simulate_returns(199, seed=5)
    obs = bf.observations(returns)
    cfg = bf.make_config(tmp_path, with_covariate=False, restarts=1)
    data = Preparer(cfg.data).fit(obs, [], train_end=obs.times[-1]).transform(obs, [])
    out = ModelStore.save(fit, tmp_path / "again.rkmodel", cfg, data, extensions=extension_payloads(cfg))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_version"] == BUNDLE_VERSION == 2
    assert manifest["package"] == "regimekit"
    assert set(manifest["extensions"]) == {"regimekit"}
    assert "profile" not in manifest
    assert sorted(p.name for p in out.iterdir() if p.is_file()) == sorted(FILES)
    payload = json.loads((out / "extensions" / "regimekit.json").read_text(encoding="utf-8"))
    assert payload["primary"]["provider"] == "csv"
    assert (out / "provenance.json").is_file()
    assert not (out / "snapshots.json").exists()
    back = ModelStore.load(out, FinanceConfig)
    assert back.loglik == fit.loglik
    assert np.array_equal(np.asarray(back.uparams), np.asarray(fit.uparams))
    assert ModelStore.load_extensions(out) == {"regimekit": payload}


def test_e10_t07a_load_extensions_of_a_v1_bundle_gives_the_snapshot_metadata() -> None:
    """A version-1 bundle's `snapshots.json` is the `regimekit` extension payload."""
    payload = ModelStore.load_extensions(FIXTURE / "v0_2_0.rkmodel")
    assert set(payload) == {"regimekit"}
    assert "snapshot_ids" in payload["regimekit"]
    assert payload["regimekit"]["primary"]["provider"] == "csv"


def test_e10_t07a_a_missing_extension_package_is_a_persistence_error_naming_it(tmp_path: Path) -> None:
    """AC4: a manifest that lists an extension that cannot be imported -> `PersistenceError` naming it."""
    fit = ModelStore.load(FIXTURE / "v0_2_0.rkmodel", FinanceConfig)
    cfg = bf.make_config(tmp_path, with_covariate=False, restarts=1)
    obs = bf.observations(bf.simulate_returns(199, seed=5))
    data = Preparer(cfg.data).fit(obs, [], train_end=obs.times[-1]).transform(obs, [])
    out = ModelStore.save(
        fit, tmp_path / "ext.rkmodel", cfg, data, extensions={"no_such_extension_pkg": {"a": 1}}
    )
    with pytest.raises(PersistenceError, match="no_such_extension_pkg") as info:
        ModelStore.load(out, FinanceConfig)
    assert info.value.path == "no_such_extension_pkg"


def test_e10_t07a_the_suffix_is_a_store_option() -> None:
    """`ModelStore(suffix=".hsmm")` names bundle directories with its suffix; the default is `.rkmodel`."""
    assert ModelStore().bundle_path("runs", "m").name == "m.rkmodel"
    assert ModelStore(suffix=".hsmm").bundle_path("runs", "m") == Path("runs") / "m.hsmm"


def test_e10_t07a_saving_over_an_old_bundle_removes_its_stale_files(tmp_path: Path) -> None:
    """Re-saving into a version-1 bundle directory leaves a clean version-2 bundle."""
    target = tmp_path / "old.rkmodel"
    shutil.copytree(FIXTURE / "v0_2_0.rkmodel", target)
    fit = ModelStore.load(target, FinanceConfig)
    cfg = bf.make_config(tmp_path, with_covariate=False, restarts=1)
    obs = bf.observations(bf.simulate_returns(199, seed=5))
    data = Preparer(cfg.data).fit(obs, [], train_end=obs.times[-1]).transform(obs, [])
    ModelStore.save(fit, target, cfg, data)
    assert sorted(p.name for p in target.iterdir() if p.is_file()) == sorted(FILES)
    assert jnp.asarray(ModelStore.load(target, FinanceConfig).uparams).dtype == jnp.float64
