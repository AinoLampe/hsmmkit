"""Temporary `params` + `spec` persistence for `decode`/`forecast` (E6-T02).

Stands in for the real bundle format until `E8-T01` (`persistence/store.py`, safetensors, ADR-0011):
this ticket's own text says a "`--model` pointing at a `params.safetensors`+`spec.json` pair written
by a temporary helper in this ticket" -- CLAUDE.md §4 invariant 6 forbids a new runtime dependency
without an ADR, and `safetensors` is ADR-0011, owned by E8-T01, so this helper writes `params.npz`
(plain NumPy, already a dependency) instead of literally using the safetensors package; `spec.json` is
the resolved `ProjectConfig`, re-parsed through `model.build.build_model_spec` on load rather than
trying to serialize arbitrary `EmissionFamily`/`DurationFamily`/`TransitionModel` instances directly.

CLAUDE.md §7: E8-T01 replaces this module and removes it (its own AC4: "`decode`/`forecast` switched
to this loader; their tests pass unchanged").
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd

from regimekit.config.schema import ProjectConfig
from regimekit.estimation.results import FitResult
from regimekit.model.build import build_model_spec
from regimekit.model.params.manager import ParamManager
from regimekit.utils.errors import PersistenceError

if TYPE_CHECKING:
    from regimekit.data.contracts import PreparedDataset

_PARAMS_FILE = "params.npz"
_SPEC_FILE = "spec.json"


def write_temp_model(fit: FitResult, cfg: ProjectConfig, path: Path | str) -> Path:
    """Write `fit.uparams` and the resolved `cfg` to `path` (created if needed)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / _PARAMS_FILE, u=np.asarray(fit.uparams))
    (out / _SPEC_FILE).write_text(cfg.model_dump_json(), encoding="utf-8")
    return out


def load_temp_model(path: Path | str, data: PreparedDataset) -> FitResult:
    """Read back a `write_temp_model` directory as a `FitResult` (`decode`'s `--model DIR`).

    Rebuilds `ModelSpec` via `model.build.build_model_spec(cfg, data)` from the stored resolved
    config, then `ParamManager.unflatten` turns the stored `u` into `theta`. The result's
    `objective`/`loglik`/`restarts`/`diagnostics` are placeholders (`decode` does not read them; the
    real bundle format restores them properly).

    Raises:
        PersistenceError: `path` is not a `write_temp_model` directory.
    """
    src = Path(path)
    spec_path, params_path = src / _SPEC_FILE, src / _PARAMS_FILE
    if not spec_path.exists() or not params_path.exists():
        raise PersistenceError(
            f"{src} is not a temporary model directory (missing {_SPEC_FILE}/{_PARAMS_FILE})",
            path=str(src),
        )
    cfg = ProjectConfig.model_validate_json(spec_path.read_text(encoding="utf-8"))
    spec = build_model_spec(cfg, data)
    manager = ParamManager(spec.param_specs)
    with np.load(params_path) as npz:
        u = jnp.asarray(npz["u"])
    theta = manager.unflatten(u)
    return FitResult(
        estimator="map",
        params=theta,
        uparams=u,
        objective=float("nan"),
        loglik=float("nan"),
        restarts=pd.DataFrame(
            columns=["restart", "init_strategy", "objective", "loglik", "n_iter", "converged", "wall_s"]
        ),
        converged=True,
        n_iter=0,
        std_errors=None,
        diagnostics={},
        spec=spec,
        prep_stats=data.prep_stats,
        config_resolved=cfg.model_dump(mode="json"),
    )
