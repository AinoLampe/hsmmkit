"""Regenerate the synthetic `gauss.*` source fixtures (E10-T04a); run it with `uv run python`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
N = 40


def frame() -> pd.DataFrame:
    """Two observed series (`y1`, `y2`), one covariate (`x`), timestamps (`t`) and sequence ids (`seq`)."""
    rng = np.random.default_rng(20241)
    return pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=N, freq="D").strftime("%Y-%m-%d"),
            "y1": rng.normal(0.0, 1.0, N),
            "y2": rng.normal(1.0, 2.0, N),
            "x": rng.normal(size=N),
            "seq": np.repeat([0, 1], N // 2),
        }
    )


if __name__ == "__main__":
    df = frame()
    df.to_csv(HERE / "gauss.csv", index=False, float_format="%.12g")
    pd.read_csv(HERE / "gauss.csv").assign(t=lambda d: pd.to_datetime(d["t"])).to_parquet(
        HERE / "gauss.parquet", index=False
    )
    csv = pd.read_csv(HERE / "gauss.csv")
    np.savez(
        HERE / "gauss.npz",
        observations=csv[["y1", "y2"]].to_numpy(),
        times=pd.to_datetime(csv["t"]).to_numpy().astype("datetime64[ns]"),
        covariates=csv[["x"]].to_numpy(),
        covariate_names=np.array(["x"]),
        sequence_ids=csv["seq"].to_numpy(),
    )
