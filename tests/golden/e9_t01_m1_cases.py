"""M=1 golden cases for E9-T01 AC2 (not collected): model assembly and the numbers recorded from them.

`record()` was run on `main` before any E9-T01 code change; `m1_golden.json` holds its output with floats
as `float.hex()` strings (exact). The test compares recomputed values against it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tests.helpers.objective_fixtures import Case, make_case

GOLDEN_PATH = Path(__file__).with_name("m1_golden.json")

# (emission, duration, order, covariates-on-emission); every case has a masked block at rows 7-9 and 20.
CASES: dict[str, tuple[str, str, int, bool]] = {
    "gbm_geometric_o1": ("gbm", "geometric", 1, False),
    "gbm_negbin_o2": ("gbm", "negbin", 2, False),
    "student_t_negbin_o1": ("student_t", "negbin", 1, False),
    "gbm_negbin_emission_cov": ("gbm", "negbin", 1, True),
    "gbm_gamma_explicit": ("gbm", "gamma", 1, False),
}
MASKED_ROWS = (7, 8, 9, 20)


def build(name: str) -> Case:
    """The golden case `name`: `make_case` plus a few fully masked rows."""
    emission, duration, order, cov_em = CASES[name]
    case = make_case(emission, duration, order, cov_em=cov_em, t_len=60, seed=11)
    mask = np.ones(case.data.mask.shape, dtype=bool)
    mask[list(MASKED_ROWS)] = False
    return dataclasses.replace(case, data=dataclasses.replace(case.data, mask=jnp.asarray(mask)))


def compute(name: str) -> dict[str, object]:
    """`loglik` and the `forward_backward` smoothed posterior of case `name` at its starting point."""
    case = build(name)
    theta = case.manager.unflatten(case.u0)
    post = case.engine.forward_backward(theta, case.spec, case.space, case.data)
    smoothed = np.asarray(post.smoothed, dtype=np.float64)
    return {
        "loglik": float(post.loglik),
        "smoothed_shape": list(smoothed.shape),
        "smoothed_sum": float(smoothed.sum()),
        "smoothed_hex": [float(v).hex() for v in smoothed.ravel()],
    }


def record() -> None:
    """Write `m1_golden.json` (run once, on `main`, before the E9-T01 code change)."""
    out = {}
    for name in CASES:
        res = compute(name)
        res["loglik"] = float(res["loglik"]).hex()  # type: ignore[arg-type]
        res["smoothed_sum"] = float(res["smoothed_sum"]).hex()  # type: ignore[arg-type]
        out[name] = res
    GOLDEN_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    record()
