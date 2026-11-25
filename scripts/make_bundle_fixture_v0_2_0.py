"""Write the `tests/fixtures/bundles/v0_2_0.rkmodel/` bundle fixture (E10-T07a).

Run it with the code of tag `v0.2.0` (a worktree of the tag on `PYTHONPATH`), so that the bundle is a genuine
`bundle_version 1` bundle written by `ModelStore.save` of that release:

    git worktree add <dir> v0.2.0
    PYTHONPATH=<dir>/src:<dir> uv run --no-sync python scripts/make_bundle_fixture_v0_2_0.py <out>

It fits the small synthetic model of `tests/integration/test_e8_t01_bundle.py` (geometric durations, no
covariates) and writes the bundle plus `expected.json` (the log-likelihood and the unconstrained parameters).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from regimekit.data.prepare import Preparer
from regimekit.estimation.map import fit
from regimekit.model.build import build_model_spec
from regimekit.persistence.store import ModelStore
from tests.helpers import backtest_fixtures as bf


def main(out: Path) -> None:
    """Fit the model and write the bundle and `expected.json` into `out`."""
    returns = bf.simulate_returns(199, seed=5)
    obs = bf.observations(returns)
    (out / "work").mkdir(parents=True, exist_ok=True)
    cfg = bf.make_config(out / "work", with_covariate=False, restarts=1)
    cfg.estimation.max_iter = 40
    data = Preparer(cfg.data).fit(obs, [], train_end=obs.times[-1]).transform(obs, [])
    spec = build_model_spec(cfg, data)
    result = fit(spec, data, cfg.estimation, seed=7)
    ModelStore.save(result, out / "v0_2_0.rkmodel", cfg, data)
    expected = {"loglik": result.loglik, "uparams": np.asarray(result.uparams).tolist()}
    (out / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
