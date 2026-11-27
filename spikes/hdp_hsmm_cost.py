"""E13-T09 spike: cost of one explicit-engine FFBS sweep at HDP-HSMM truncation sizes (not a package module).

Usage: `JAX_PLATFORMS=cpu uv run python spikes/hdp_hsmm_cost.py L D_MAX T` (for example `20 200 10000`).
Prints the first-call (compile + run) and warm times of `ffbs_explicit` and of the explicit E-step, and the
peak RSS. The numbers feed ADR-0027 (`docs/adr/0027-hdp-hsmm.md`).
"""

from __future__ import annotations

import resource
import sys
import tempfile
import time
import warnings
from pathlib import Path

import jax
import numpy as np

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.estep_explicit import explicit_estep
from hsmmkit.engine.ffbs import ffbs_explicit
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_explicit
from tests.helpers import gem_models


def _timed(fn):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202
    started = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return time.perf_counter() - started


def main(n_states: int, d_max: int, t_len: int) -> None:
    """Time FFBS and the E-step of a random `n_states`-regime model on `t_len` normal observations."""
    warnings.simplefilter("ignore")
    spec = gem_models.build(
        Path(tempfile.mkdtemp()),
        n_states,
        d_max,
        mean_sojourn=20.0,
        means=list(np.linspace(-2, 2, n_states)),
        scales=[0.8] * n_states,
    )
    manager = ParamManager(spec.param_specs)
    theta = dict(manager.unflatten(manager.init_u()))
    space = build_explicit(spec)
    data = gem_models.dataset(np.random.default_rng(0).normal(size=(t_len, 1)))
    pb = SequenceBatch.of(data).padded()
    ffbs = [_timed(lambda k=k: ffbs_explicit(jax.random.key(k), theta, spec, space, pb, 1)) for k in (0, 1)]
    estep = [_timed(lambda: explicit_estep(None, theta, spec, space, pb)) for _ in (0, 1)]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"L={n_states} D_max={d_max} T={t_len}: ffbs first {ffbs[0]:.1f}s warm {ffbs[1]:.2f}s | "
        f"estep first {estep[0]:.1f}s warm {estep[1]:.2f}s | maxrss {rss:.0f} MB"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
