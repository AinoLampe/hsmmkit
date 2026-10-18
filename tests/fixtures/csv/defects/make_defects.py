"""Regenerate `DEFECTS.csv`: synthetic daily prices with injected data-quality defects (E2-T08).

Defects: weekday gap 2023-01-17..2023-01-19 (3 business days missing), duplicated 2023-02-01,
zero price on 2023-02-10, a +10 % (10 sigma for 1 % daily vol) jump into 2023-03-01.
Run: `uv run python tests/fixtures/csv/defects/make_defects.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Write the fixture next to this script."""
    rng = np.random.default_rng(2023)
    dates = pd.bdate_range("2023-01-02", periods=60)
    log_ret = rng.normal(0.0, 0.01, len(dates))
    jump = dates.get_loc(pd.Timestamp("2023-03-01"))
    log_ret[jump] = 0.10
    close = 100.0 * np.exp(np.cumsum(log_ret))
    frame = pd.DataFrame({"time": dates.strftime("%Y-%m-%d"), "close": close.round(6)})
    frame["adj_close"] = frame["close"]
    frame.loc[frame["time"] == "2023-02-10", ["close", "adj_close"]] = 0.0
    frame = frame[~frame["time"].isin(["2023-01-17", "2023-01-18", "2023-01-19"])]
    dup = frame[frame["time"] == "2023-02-01"].copy()
    dup[["close", "adj_close"]] = (dup[["close", "adj_close"]] * 1.001).round(6)
    frame = pd.concat([frame, dup]).sort_values("time", kind="stable")
    frame.to_csv(Path(__file__).with_name("DEFECTS.csv"), index=False)


if __name__ == "__main__":
    main()
