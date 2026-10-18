"""`ModelStore`: the `<name>.rkmodel/` bundle (E8-T01, ADR-0011).

Concept refs: §5.8 persistence. A bundle persists a `FitResult` so that it reloads without data and
reproduces its log-likelihood bitwise on the same backend:

- `manifest.json`: `bundle_version`, regimekit/jax versions, backend, `created_at`, `seed`, `profile`,
  `schema_version`
- `config.resolved.yaml`: the full resolved configuration
- `spec.json`: the `ModelSpec` structure (family names and classes, `n_states`, `order`, `d_max`, `engine`,
  `initial`, covariate names), the data-derived build inputs and the expected parameter names and shapes
- `params.safetensors`: constrained `theta` (key = parameter name) and the unconstrained `u` (key
  `__u__`), float64
- `prep_stats.json`: `PrepStats`
- `snapshots.json`: snapshot ids and provider metadata
- `fit_report.json`: objective, loglik, convergence, restarts table, diagnostics, standard errors

`load` rebuilds the `ModelSpec` from the resolved configuration and the stored build inputs through the
family registries (`model.build.build_model_spec_from`), so no dataset is needed, then checks that the
stored `theta` has the expected names and shapes and equals `ParamManager.unflatten(u)` to 1e-12.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from safetensors.numpy import load_file, save_file

from regimekit.config.schema import ProjectConfig
from regimekit.config.yaml_io import dump_resolved, load_config
from regimekit.data.prepare import prep_stats_from_json, prep_stats_to_json
from regimekit.estimation.results import FitResult
from regimekit.model.build import BuildInputs, build_model_spec_from
from regimekit.model.params.manager import ParamManager
from regimekit.utils.errors import ConfigError, PersistenceError

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset
    from regimekit.model.spec import ModelSpec

BUNDLE_VERSION = 1
REGISTERED_PROFILES = frozenset({"finance"})  # `core` has no execution path in 0.1.0 (overview section 8)
U_KEY = "__u__"
THETA_TOLERANCE = 1e-12

MANIFEST = "manifest.json"
CONFIG = "config.resolved.yaml"
SPEC = "spec.json"
PARAMS = "params.safetensors"
PREP_STATS = "prep_stats.json"
SNAPSHOTS = "snapshots.json"
FIT_REPORT = "fit_report.json"
FILES = (MANIFEST, CONFIG, SPEC, PARAMS, PREP_STATS, SNAPSHOTS, FIT_REPORT)

_RESTART_COLUMNS = ["restart", "init_strategy", "objective", "loglik", "n_iter", "converged", "wall_s"]


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    """JSON-ready copy: arrays to lists, numpy/jax scalars to Python numbers, tuples to lists."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray | jax.Array):
        return _jsonable(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _package_version() -> str:
    try:
        return version("regimekit")
    except PackageNotFoundError:  # a source tree without an installed distribution
        return "unknown"


def _read_json(path: Path) -> Any:  # noqa: ANN401
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PersistenceError(
            f"cannot read {path.name} of the bundle {path.parent}: {exc}", path=str(path)
        ) from exc


class ModelStore:
    """Save and load `<name>.rkmodel/` bundles (see the module docstring)."""

    @staticmethod
    def save(
        fit: FitResult,
        path: Path | str,
        cfg: ProjectConfig,
        data: PreparedDataset,
        *,
        seed: int | None = None,
    ) -> Path:
        """Write `fit` to the bundle directory `path` (created; files of an older bundle are replaced).

        `cfg` is the resolved configuration the model was built from and `data` the dataset it was fitted
        on: they supply what a `FitResult` does not carry (the configuration, the covariate name split
        per entry point, `nominal_dt`, the snapshot ids). `seed` defaults to `cfg.project.seed`.

        Raises:
            PersistenceError: the directory cannot be written.
        """
        out = Path(path)
        spec = fit.spec
        manager = ParamManager(spec.param_specs)
        theta = {
            name: np.ascontiguousarray(np.asarray(v, dtype=np.float64)) for name, v in fit.params.items()
        }
        arrays = {**theta, U_KEY: np.ascontiguousarray(np.asarray(fit.uparams, dtype=np.float64))}
        inputs = BuildInputs.from_data(data)
        try:  # the bundle must be loadable: cfg + data have to rebuild the model that was fitted
            _check_param_layout(
                [{"name": s.name, "shape": list(s.shape)} for s in spec.param_specs],
                build_model_spec_from(cfg, inputs),
                out,
            )
        except ConfigError as exc:
            raise PersistenceError(
                f"cfg and data do not rebuild the fitted model: {exc}", path=str(out)
            ) from exc
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "regimekit_version": _package_version(),
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "created_at": datetime.now(UTC).isoformat(),
            "seed": cfg.project.seed if seed is None else seed,
            "profile": cfg.profile,
            "schema_version": cfg.schema_version,
        }
        spec_json = {
            "n_states": spec.n_states,
            "n_series": spec.n_series,
            "order": spec.order,
            "d_max": spec.d_max,
            "engine": spec.engine,
            "initial": spec.initial,
            "duration_conditioning": spec.duration_conditioning,
            "emission": {"family": cfg.model.emission.family, "class": type(spec.emission).__name__},
            "duration": {
                "family": cfg.model.duration.family,
                "class": type(spec.duration).__name__,
                "r": cfg.model.duration.r,
            },
            "transition": {"mode": cfg.model.transition.mode, "class": type(spec.transition).__name__},
            "build_inputs": {
                "n_series": inputs.n_series,
                "cov_names": {k: list(v) for k, v in inputs.cov_names.items()},
                "clock": inputs.clock,
                "nominal_dt": inputs.nominal_dt,
            },
            "params": [{"name": s.name, "shape": list(s.shape)} for s in spec.param_specs],
            "n_free": manager.n_free,
        }
        snapshots = {
            "snapshot_ids": list(data.snapshot_ids),
            "primary": {
                "provider": cfg.data.primary.provider,
                "symbols": list(cfg.data.primary.symbols),
                "interval": cfg.data.primary.interval,
                "adjusted": cfg.data.primary.adjusted,
            },
            "covariates": [
                {"name": c.name, "provider": c.provider, "series": c.series} for c in cfg.data.covariates
            ],
        }
        report = {
            "objective": fit.objective,
            "loglik": fit.loglik,
            "converged": fit.converged,
            "n_iter": fit.n_iter,
            "restarts": _jsonable(fit.restarts.to_dict(orient="records")),
            "diagnostics": _jsonable(fit.diagnostics),
            "std_errors": None if fit.std_errors is None else _jsonable(fit.std_errors),
        }
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / MANIFEST).unlink(missing_ok=True)  # written last: an interrupted save is not loadable
            dump_resolved(cfg, out / CONFIG)
            (out / SPEC).write_text(json.dumps(spec_json, indent=2), encoding="utf-8")
            save_file(arrays, str(out / PARAMS))
            (out / PREP_STATS).write_text(prep_stats_to_json(fit.prep_stats), encoding="utf-8")
            (out / SNAPSHOTS).write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
            (out / FIT_REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
            (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except OSError as exc:
            raise PersistenceError(f"cannot write the model bundle {out}: {exc}", path=str(out)) from exc
        return out

    @staticmethod
    def load(path: Path | str) -> FitResult:
        """Read a bundle back as a `FitResult` (see `_load`); malformed contents are `PersistenceError`s."""
        try:
            return ModelStore._load(path)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise PersistenceError(f"{path} is a malformed model bundle: {exc!r}", path=str(path)) from exc

    @staticmethod
    def _load(path: Path | str) -> FitResult:
        """Read a bundle back as a `FitResult` (no dataset needed; `loglik` is checked by the caller).

        Raises:
            PersistenceError: a missing or unreadable file, a `bundle_version` newer than this release, a
                `profile` that is not registered, a configuration that no longer builds, or parameters
                whose names or shapes disagree with the model (the message names the parameter).
        """
        src = Path(path)
        if not src.is_dir():
            raise PersistenceError(f"{src} is not a model bundle directory", path=str(src))
        missing = [name for name in FILES if not (src / name).is_file()]
        if missing:
            raise PersistenceError(
                f"{src} is not a complete model bundle (missing {', '.join(missing)})", path=str(src)
            )

        manifest = _read_json(src / MANIFEST)
        bundle_version = manifest.get("bundle_version")
        if not isinstance(bundle_version, int) or bundle_version > BUNDLE_VERSION:
            raise PersistenceError(
                f"{src} is bundle version {bundle_version!r}, newer than the supported {BUNDLE_VERSION}: "
                "upgrade regimekit to load it",
                path=str(src / MANIFEST),
            )
        profile = manifest.get("profile")
        if profile not in REGISTERED_PROFILES:
            raise PersistenceError(
                f"{src} was written for the profile {profile!r}, which is not registered "
                f"(known: {sorted(REGISTERED_PROFILES)})",
                path=str(src / MANIFEST),
            )

        try:
            cfg = load_config(src / CONFIG)
        except ConfigError as exc:
            raise PersistenceError(
                f"the configuration stored in {src} is invalid: {exc}", path=str(src / CONFIG)
            ) from exc
        spec_json = _read_json(src / SPEC)
        bi = spec_json["build_inputs"]
        inputs = BuildInputs(
            int(bi["n_series"]),
            {k: tuple(v) for k, v in bi["cov_names"].items()},
            bi["clock"],
            float(bi["nominal_dt"]),
        )
        try:
            spec = build_model_spec_from(cfg, inputs)
        except ConfigError as exc:
            raise PersistenceError(f"cannot rebuild the model of {src}: {exc}", path=str(src / SPEC)) from exc
        manager = ParamManager(spec.param_specs)
        _check_param_layout(spec_json["params"], spec, src)

        try:
            stored = load_file(str(src / PARAMS))
        except Exception as exc:  # safetensors raises its own error types for corrupt files
            raise PersistenceError(f"cannot read {PARAMS} of {src}: {exc}", path=str(src / PARAMS)) from exc
        u_np = stored.pop(U_KEY, None)
        if u_np is None or u_np.shape != (manager.n_free,):
            got = None if u_np is None else tuple(u_np.shape)
            raise PersistenceError(
                f"{PARAMS} of {src}: {U_KEY} has shape {got}, expected ({manager.n_free},)",
                path=str(src / PARAMS),
            )
        u = jnp.asarray(u_np)
        reference = manager.unflatten(u)
        _check_theta(stored, reference, src)
        theta = {name: jnp.asarray(stored[name]) for name in reference}

        report = _read_json(src / FIT_REPORT)
        restarts = pd.DataFrame(report["restarts"], columns=_RESTART_COLUMNS)
        std_errors = report["std_errors"]
        return FitResult(
            estimator="map",
            params=theta,
            uparams=u,
            objective=float(report["objective"]),
            loglik=float(report["loglik"]),
            restarts=restarts,
            converged=bool(report["converged"]),
            n_iter=int(report["n_iter"]),
            std_errors=None if std_errors is None else {k: jnp.asarray(v) for k, v in std_errors.items()},
            diagnostics=report["diagnostics"],
            spec=spec,
            prep_stats=prep_stats_from_json((src / PREP_STATS).read_text(encoding="utf-8")),
            config_resolved=cfg.model_dump(mode="json"),
        )


def _check_param_layout(saved: list[dict[str, Any]], spec: ModelSpec, src: Path) -> None:
    """The parameter names and shapes recorded at save time must be those of the rebuilt model."""
    expected = {s.name: tuple(s.shape) for s in spec.param_specs}
    recorded = {str(p["name"]): tuple(p["shape"]) for p in saved}
    for name in sorted(set(expected) | set(recorded)):
        if expected.get(name) != recorded.get(name):
            raise PersistenceError(
                f"parameter {name!r} of {src}: the bundle records shape {recorded.get(name)}, "
                f"the rebuilt model has {expected.get(name)}",
                path=name,
            )


def _check_theta(stored: dict[str, np.ndarray], reference: dict[str, Any], src: Path) -> None:
    """Names and shapes of the stored `theta` match the model, and it equals `unflatten(u)` to 1e-12."""
    for name, ref in reference.items():
        if name not in stored:
            raise PersistenceError(f"parameter {name!r} is missing from {PARAMS} of {src}", path=name)
        got = stored[name]
        if got.shape != tuple(ref.shape):
            raise PersistenceError(
                f"parameter {name!r} of {src}: stored shape {tuple(got.shape)}, expected {tuple(ref.shape)}",
                path=name,
            )
        # NaN in both is agreement (a crafted non-finite model must reach the engine's NumericalError)
        ref_np = np.asarray(ref)
        diff = np.abs(got - ref_np) / (1.0 + np.abs(ref_np))  # absolute near 0, relative for large values
        gap = float(np.max(np.where(np.isnan(got) & np.isnan(ref_np), 0.0, diff))) if got.size else 0.0
        if not gap <= THETA_TOLERANCE:
            raise PersistenceError(
                f"parameter {name!r} of {src} differs from unflatten(u) by {gap:.3g} (> {THETA_TOLERANCE:g})",
                path=name,
            )
    extra = sorted(set(stored) - set(reference))
    if extra:
        raise PersistenceError(f"{PARAMS} of {src} holds unknown parameter(s) {extra}", path=extra[0])
