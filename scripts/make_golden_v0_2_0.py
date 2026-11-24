"""Generate / recompute the v0.2.0 golden values (E10-T01).

`compute(work)` runs the Appendix A pipeline offline on the synthetic csv fixtures (`examples/synthetic.py`,
the data of E8-T03) and returns the arrays to compare. `python scripts/make_golden_v0_2_0.py` writes them to
`tests/golden/v0_2_0/`. Regenerating is allowed only after a toolchain change (JAX/NumPy/backend), with the
owner's approval, and must be recorded in `manifest.json` and `spec/STATUS.md` (CLAUDE.md section 3.9).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(os.environ.get("REGIMEKIT_GOLDEN_DIR", ROOT / "tests" / "golden" / "v0_2_0"))
EXAMPLE = ROOT / "examples" / "configs" / "dax_3state_gbm_nb.yaml"
HORIZON, N_SIM, SEED = 20, 2000, 42
# Two walk-forward folds on the 3-year synthetic sample, reduced simulation size.
OVERRIDES: dict[str, Any] = {
    "evaluation": {"backtest": {"refit_every_steps": 189, "n_sim": 100}},
}
FILES = ("prepared.npz", "fit.npz", "posterior.npz", "forecast.npz", "backtest_metrics.parquet")


def compute(work: Path) -> dict[str, dict[str, np.ndarray] | pd.DataFrame]:
    """Run the pipeline in `work` and return the five golden groups."""
    sys.path.insert(0, str(ROOT))
    from examples.synthetic import write_offline_project
    from regimekit.project import Project

    config = write_offline_project(work, EXAMPLE, overrides=OVERRIDES)
    project = Project.from_yaml(config, out_dir=work / "runs", offline=False)
    data = project.fetch()
    fit = project.fit(data)
    post = project.decode(fit, data)
    fc = project.forecast(fit, data, horizon=HORIZON, n_sim=N_SIM)
    bt = project.backtest()

    prepared: dict[str, np.ndarray] = {
        "observations": np.asarray(data.observations),
        "dt": np.asarray(data.dt),
        "mask": np.asarray(data.mask),
    }
    for name in ("cov_transition", "cov_duration", "cov_emission"):
        value = getattr(data, name)
        if value is not None:
            prepared[name] = np.asarray(value)
    return {
        "prepared.npz": prepared,
        "fit.npz": {
            "uparams": np.asarray(fit.uparams),
            "loglik": np.asarray(fit.loglik, dtype=np.float64),
            "objective": np.asarray(fit.objective, dtype=np.float64),
        },
        "posterior.npz": {
            "filtered": np.asarray(post.filtered),
            "smoothed": np.asarray(post.smoothed),
            "viterbi": np.asarray(post.viterbi),
        },
        "forecast.npz": {"samples": np.asarray(fc.samples)},
        "backtest_metrics.parquet": bt.metrics.reset_index(drop=True),
    }


def load_stored(directory: Path | None = None) -> dict[str, dict[str, np.ndarray] | pd.DataFrame]:
    """The golden values in `directory` (default: the committed ones)."""
    directory = directory or GOLDEN_DIR
    out: dict[str, dict[str, np.ndarray] | pd.DataFrame] = {}
    for name in FILES:
        path = directory / name
        if name.endswith(".npz"):
            with np.load(path) as z:
                out[name] = {k: z[k] for k in z.files}
        else:
            out[name] = pd.read_parquet(path)
    return out


def differences(
    stored: dict[str, dict[str, np.ndarray] | pd.DataFrame],
    fresh: dict[str, dict[str, np.ndarray] | pd.DataFrame],
) -> list[str]:
    """Names of the golden entries that are not bitwise equal."""
    bad: list[str] = []
    for name in FILES:
        a, b = stored[name], fresh[name]
        if isinstance(a, pd.DataFrame):
            assert isinstance(b, pd.DataFrame)
            if not (a.shape == b.shape and list(a.columns) == list(b.columns) and a.equals(b)):
                bad.append(name)
            continue
        assert isinstance(b, dict)
        if set(a) != set(b):
            bad.append(f"{name}: keys {sorted(a)} != {sorted(b)}")
            continue
        bad += [
            f"{name}:{k}"
            for k in a
            if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])
        ]
    return bad


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _machine() -> str:
    """CPU model and kernel, for diagnosing cross-machine differences (not compared)."""
    model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{model}; {platform.platform()}"


def write_golden(fresh: dict[str, dict[str, np.ndarray] | pd.DataFrame], note: str) -> None:
    """Write the golden files and the manifest."""
    import jax
    import jaxlib

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = GOLDEN_DIR / "manifest.json"
    notes: list[str] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))["notes"] if manifest_path.is_file() else []
    )
    for name, value in fresh.items():
        if isinstance(value, pd.DataFrame):
            value.to_parquet(GOLDEN_DIR / name, index=False)
        else:
            np.savez(GOLDEN_DIR / name, **value)
    manifest = {
        "baseline": "regimekit v0.2.0 (E10-T01)",
        "git_sha": _git_sha(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "horizon": HORIZON,
        "n_sim": N_SIM,
        "seed": SEED,
        "machine": _machine(),
        "notes": [*notes, note],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Command line entry: `python scripts/make_golden_v0_2_0.py --note "..."`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", help="why the values are regenerated (writes to REGIMEKIT_GOLDEN_DIR)")
    parser.add_argument(
        "--compare",
        type=Path,
        help="do not write: recompute and compare bitwise with the golden files in this directory "
        "(single-machine check: write a baseline from the merge base with REGIMEKIT_GOLDEN_DIR first)",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        fresh = compute(Path(tmp))
    if args.compare is not None:
        bad = differences(load_stored(args.compare), fresh)
        print("DIFFERENCES:", bad)
        raise SystemExit(1 if bad else 0)
    if not args.note:
        parser.error("--note is required unless --compare is given")
    write_golden(fresh, args.note)


if __name__ == "__main__":
    main()
