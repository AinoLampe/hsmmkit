"""E8-T01: `ModelStore` bundle -- lossless round trip, file layout, version/profile/shape guards."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from regimekit.config.schema import ProjectConfig
from regimekit.config.yaml_io import load_config
from regimekit.data.contracts import PreparedDataset
from regimekit.data.prepare import Preparer
from regimekit.engine.expanded_hmm import ExpandedHMMEngine
from regimekit.estimation.map import fit
from regimekit.estimation.objective import canonical_data
from regimekit.estimation.results import FitResult
from regimekit.finance import FinanceConfig
from regimekit.model.build import build_model_spec
from regimekit.model.params.manager import ParamManager
from regimekit.model.spec import ModelSpec
from regimekit.persistence.store import BUNDLE_VERSION, FILES, U_KEY, ModelStore
from regimekit.statespace.builders import build_expanded
from regimekit.utils.errors import PersistenceError
from tests.helpers import backtest_fixtures as bf

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)


@dataclass
class Fitted:
    """One fitted model with the configuration and dataset it came from."""

    fit: FitResult
    cfg: ProjectConfig
    data: PreparedDataset


def _fitted(tmp_path: Path, *, covariate: bool = False, negbin: bool = False) -> Fitted:
    returns = bf.simulate_returns(199, seed=5)
    obs = bf.observations(returns)
    covs = [bf.covariate(obs, np.concatenate([[0.0], returns]))] if covariate else []
    cfg = bf.make_config(tmp_path, with_covariate=covariate, restarts=1)
    cfg.estimation.max_iter = 40
    if negbin:
        cfg.model.duration.family = "negbin"
        cfg.model.duration.r = [2, 2]
    data = Preparer(cfg.data).fit(obs, covs, train_end=obs.times[-1]).transform(obs, covs)
    spec = build_model_spec(cfg, data)
    return Fitted(fit(spec, data, cfg.estimation, seed=7), cfg, data)


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Fitted:
    """A fitted geometric-duration model (no covariates)."""
    return _fitted(tmp_path_factory.mktemp("plain"))


@pytest.fixture
def bundle(plain: Fitted, tmp_path: Path) -> Path:
    """A fresh bundle of the plain model."""
    return ModelStore.save(plain.fit, tmp_path / "m.rkmodel", plain.cfg, plain.data)


def _assert_same_structure(a: ModelSpec, b: ModelSpec) -> None:
    """Equal scalar fields, family types and parameter specs (`ModelSpec` itself has no usable `==`)."""
    for field in ("n_states", "n_series", "order", "d_max", "engine", "initial", "duration_conditioning"):
        assert getattr(a, field) == getattr(b, field), field
    for family in ("emission", "duration", "transition"):
        assert type(getattr(a, family)) is type(getattr(b, family)), family
    assert [(p.name, p.shape, p.domain, p.mode) for p in a.param_specs] == [
        (p.name, p.shape, p.domain, p.mode) for p in b.param_specs
    ]
    for pa, pb in zip(a.param_specs, b.param_specs, strict=True):
        assert np.array_equal(np.asarray(pa.init), np.asarray(pb.init)), pa.name


def _loglik(fitted: FitResult, data: PreparedDataset, *, canonical: bool = True) -> float:
    space = build_expanded(fitted.spec)
    d = canonical_data(data) if canonical else data
    return float(ExpandedHMMEngine().loglik(fitted.params, fitted.spec, space, d))


# --- AC1 -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["plain", "covariate", "negbin"])
def test_e8_t01_round_trip_is_bitwise_for_params_and_loglik(
    variant: str, plain: Fitted, tmp_path: Path
) -> None:
    """AC1: `load(save(fit))` has bitwise equal `params`/`u`/`loglik` and reproduces `engine.loglik`."""
    source = (
        plain
        if variant == "plain"
        else _fitted(tmp_path, covariate=variant == "covariate", negbin=variant == "negbin")
    )
    path = ModelStore.save(source.fit, tmp_path / "m.rkmodel", source.cfg, source.data)
    back = ModelStore.load(path, FinanceConfig)

    assert set(back.params) == set(source.fit.params)
    for name, value in source.fit.params.items():
        assert np.array_equal(np.asarray(back.params[name]), np.asarray(value)), name
        assert back.params[name].dtype == value.dtype
    assert np.array_equal(np.asarray(back.uparams), np.asarray(source.fit.uparams))
    _assert_same_structure(back.spec, source.fit.spec)
    assert back.loglik == source.fit.loglik
    assert back.objective == source.fit.objective
    assert (back.converged, back.n_iter) == (source.fit.converged, source.fit.n_iter)
    assert back.prep_stats == source.fit.prep_stats
    assert back.estimator == "map"
    assert list(back.restarts.columns) == list(source.fit.restarts.columns)
    assert len(back.restarts) == len(source.fit.restarts)

    ll = _loglik(back, source.data)
    assert ll == source.fit.loglik  # bitwise, same backend
    assert ll == _loglik(source.fit, source.data, canonical=False)
    assert ParamManager(back.spec.param_specs).n_free == back.uparams.shape[0]


def test_e8_t01_load_does_not_need_data_and_rebuilds_the_covariate_split(tmp_path: Path) -> None:
    """The bundle of a covariate model stores the entry-point split, so the spec rebuilds from files alone."""
    source = _fitted(tmp_path, covariate=True)
    path = ModelStore.save(source.fit, tmp_path / "c.rkmodel", source.cfg, source.data)
    spec_json = json.loads((path / "spec.json").read_text(encoding="utf-8"))
    assert spec_json["build_inputs"]["cov_names"]["emission"] == ["signal"]
    assert any(p["name"] == "emission.gamma" for p in spec_json["params"])
    back = ModelStore.load(path, FinanceConfig)
    assert "emission.gamma" in back.params
    _assert_same_structure(back.spec, source.fit.spec)


# --- AC2 -----------------------------------------------------------------------------------------


def test_e8_t01_bundle_has_all_files_with_the_documented_keys(plain: Fitted, bundle: Path) -> None:
    """AC2: the bundle-v2 files; manifest keys incl. `package`; `__u__` in the safetensors file."""
    assert sorted(p.name for p in bundle.iterdir() if p.is_file()) == sorted(FILES)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for key in (
        "bundle_version", "package", "package_version", "extensions", "jax_version", "backend", "created_at",
        "seed", "schema_version",
    ):  # fmt: skip
        assert key in manifest, key
    assert manifest["bundle_version"] == BUNDLE_VERSION == 2
    assert "profile" not in manifest
    assert manifest["package"] == "regimekit"
    assert manifest["schema_version"] == plain.cfg.schema_version
    assert manifest["seed"] == plain.cfg.project.seed

    with safe_open(str(bundle / "params.safetensors"), framework="numpy") as f:
        keys = set(f.keys())
    assert U_KEY in keys
    assert keys == {U_KEY, *plain.fit.params}

    spec_json = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    for key in (
        "n_states",
        "order",
        "d_max",
        "engine",
        "initial",
        "emission",
        "duration",
        "build_inputs",
        "params",
    ):
        assert key in spec_json, key
    assert spec_json["emission"]["family"] == "gbm"
    assert spec_json["duration"]["class"] == "GeometricDuration"
    assert load_config(bundle / "config.resolved.yaml", FinanceConfig).model_dump() == plain.cfg.model_dump()
    snapshots = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    assert snapshots["snapshot_ids"] == list(plain.data.snapshot_ids)
    report = json.loads((bundle / "fit_report.json").read_text(encoding="utf-8"))
    assert report["loglik"] == plain.fit.loglik
    assert len(report["restarts"]) == len(plain.fit.restarts)


def test_e8_t01_the_profile_key_of_the_manifest_is_ignored(bundle: Path) -> None:
    """Bundle v2 has no `profile`; a stale key in a manifest is not an error (E10-T07a)."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile"] = "quantum"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ModelStore.load(bundle, FinanceConfig)


# --- AC3 -----------------------------------------------------------------------------------------


def test_e8_t01_a_newer_bundle_version_is_refused(bundle: Path) -> None:
    """AC3: `bundle_version` above the supported one -> a clear `PersistenceError`."""
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_version"] = BUNDLE_VERSION + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PersistenceError, match="newer"):
        ModelStore.load(bundle, FinanceConfig)


def test_e8_t01_a_parameter_with_the_wrong_shape_is_reported_by_name(bundle: Path) -> None:
    """AC3: corrupted params (shape mismatch) -> `PersistenceError` naming the parameter."""
    arrays = load_file(str(bundle / "params.safetensors"))
    arrays["emission.sigma"] = np.concatenate([arrays["emission.sigma"], [0.5]])
    save_file(arrays, str(bundle / "params.safetensors"))
    with pytest.raises(PersistenceError, match=r"emission\.sigma") as err:
        ModelStore.load(bundle, FinanceConfig)
    assert err.value.path == "emission.sigma"


def test_e8_t01_theta_that_disagrees_with_u_is_refused(bundle: Path) -> None:
    """The stored `theta` must equal `unflatten(u)` (1e-12): a tampered value is caught."""
    arrays = load_file(str(bundle / "params.safetensors"))
    arrays["emission.mu"] = arrays["emission.mu"] + 1e-6
    save_file(arrays, str(bundle / "params.safetensors"))
    with pytest.raises(PersistenceError, match=r"emission\.mu"):
        ModelStore.load(bundle, FinanceConfig)


def test_e8_t01_missing_files_and_foreign_directories_are_persistence_errors(
    bundle: Path, tmp_path: Path
) -> None:
    """A missing file, a missing `__u__`, a non-bundle directory and a missing path all raise."""
    broken = tmp_path / "broken.rkmodel"
    shutil.copytree(bundle, broken)
    (broken / "fit_report.json").unlink()
    with pytest.raises(PersistenceError, match=r"fit_report\.json"):
        ModelStore.load(broken, FinanceConfig)

    no_u = tmp_path / "no_u.rkmodel"
    shutil.copytree(bundle, no_u)
    arrays = load_file(str(no_u / "params.safetensors"))
    arrays.pop(U_KEY)
    save_file(arrays, str(no_u / "params.safetensors"))
    with pytest.raises(PersistenceError, match=U_KEY):
        ModelStore.load(no_u, FinanceConfig)

    with pytest.raises(PersistenceError):
        ModelStore.load(tmp_path, FinanceConfig)
    with pytest.raises(PersistenceError):
        ModelStore.load(tmp_path / "nope.rkmodel", FinanceConfig)


def test_e8_t01_saving_over_an_existing_bundle_replaces_it(plain: Fitted, bundle: Path) -> None:
    """A second `save` into the same directory leaves a loadable bundle."""
    ModelStore.save(plain.fit, bundle, plain.cfg, plain.data)
    assert np.array_equal(
        np.asarray(ModelStore.load(bundle, FinanceConfig).uparams), np.asarray(plain.fit.uparams)
    )
    assert jnp.asarray(ModelStore.load(bundle, FinanceConfig).uparams).dtype == jnp.float64


def test_e8_t01_nan_parameters_round_trip_so_the_engine_can_report_them(
    plain: Fitted, tmp_path: Path
) -> None:
    """A non-finite `u` is stored and reloaded as it is; only a mismatch between `theta` and `u` fails."""
    manager = ParamManager(plain.fit.spec.param_specs)
    nan_u = jnp.full_like(manager.init_u(), jnp.nan)
    nan_fit = FitResult(**{**plain.fit.__dict__, "uparams": nan_u, "params": manager.unflatten(nan_u)})
    path = ModelStore.save(nan_fit, tmp_path / "nan.rkmodel", plain.cfg, plain.data)
    back = ModelStore.load(path, FinanceConfig)
    assert bool(jnp.all(jnp.isnan(back.uparams)))


def test_e8_t01_save_refuses_a_config_that_does_not_rebuild_the_fitted_model(
    plain: Fitted, tmp_path: Path
) -> None:
    """A mismatching `cfg` fails at `save`, not later at `load`."""
    other = plain.cfg.model_copy(deep=True)
    other.model.emission.covariates.drift = []
    other.model.emission.family = "student_t"
    with pytest.raises(PersistenceError, match="do not rebuild"):
        ModelStore.save(plain.fit, tmp_path / "bad.rkmodel", other, plain.data)
    assert not (tmp_path / "bad.rkmodel" / "manifest.json").exists()


def test_e8_t01_a_malformed_bundle_file_is_a_persistence_error_not_a_key_error(bundle: Path) -> None:
    """Missing keys or wrong types inside the JSON files surface as `PersistenceError`."""
    spec_path = bundle / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    del spec["build_inputs"]
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(PersistenceError, match="malformed"):
        ModelStore.load(bundle, FinanceConfig)
    (bundle / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(PersistenceError):
        ModelStore.load(bundle, FinanceConfig)
