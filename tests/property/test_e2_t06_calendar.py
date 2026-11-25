"""E2-T06 AC4: aligned covariates never look ahead (hypothesis)."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from hsmmkit.data.calendar import align_covariate, build_grid
from hsmmkit.data.contracts import CovariateSeries, ObservationSet

HOUR_NS = 3_600 * 10**9
ORIGIN = np.datetime64("2024-01-01T00:00", "ns")


def _series(offsets_h: list[int], values: list[float]) -> CovariateSeries:
    order = np.argsort(offsets_h, kind="stable")
    times = ORIGIN + np.asarray(offsets_h, dtype=np.int64)[order] * HOUR_NS
    return CovariateSeries(
        name="z",
        times=times.astype("datetime64[ns]"),
        values=np.asarray(values, dtype=np.float64)[order],
        source="test",
        series_id="z",
        snapshot_id="snap",
        frequency="d",
    )


@st.composite
def _cases(draw: st.DrawFn) -> tuple[list[int], list[int], list[float], int, int, int, int]:
    gaps = draw(st.lists(st.integers(1, 72), min_size=2, max_size=40))
    grid_h = np.concatenate([[0], np.cumsum(gaps)]).tolist()
    span = grid_h[-1]
    n_pub = draw(st.integers(1, 40))
    pub_h = draw(st.lists(st.integers(-100, span + 100), min_size=n_pub, max_size=n_pub, unique=True))
    values = draw(
        st.lists(
            st.one_of(st.floats(-1e3, 1e3, allow_nan=False), st.just(float("nan"))),
            min_size=n_pub,
            max_size=n_pub,
        )
    )
    lag = draw(st.integers(1, 4))
    max_staleness = draw(st.integers(0, 10))
    step = draw(st.integers(0, len(gaps) - 1))  # code index t: observation (τ_t, τ_{t+1}]
    spike_offset = draw(st.integers(1, 200))
    return grid_h, pub_h, values, lag, max_staleness, step, spike_offset


@given(_cases())
def test_e2_t06_no_lookahead_after_grid_start_of_step(
    case: tuple[list[int], list[int], list[float], int, int, int, int],
) -> None:
    """AC4: prints dated after τ_t (a spike at τ_{t+1} and later) never change rows 0..t."""
    grid_h, pub_h, values, lag, max_staleness, step, spike_offset = case
    times = (ORIGIN + np.asarray(grid_h, dtype=np.int64) * HOUR_NS).astype("datetime64[ns]")
    grid = build_grid(
        ObservationSet(
            series_names=("P",),
            times=times,
            values=np.linspace(1.0, 2.0, len(times))[:, None],
            source="test",
            snapshot_id="snap",
            meta={"interval": "1h", "adjusted": True},
        ),
        "drop",
        clock="calendar",
    )
    base_values, base_staleness = align_covariate(grid, _series(pub_h, values), lag, max_staleness)

    # Spike: a print exactly at τ_{t+1} (the end of observation t), a later one, and every existing
    # print dated after τ_t overwritten. None of it may change rows 0..t.
    cutoff = grid_h[step]
    spiked = {h: (v if h <= cutoff else 1e9) for h, v in zip(pub_h, values, strict=True)}
    spiked[grid_h[step + 1]] = -1e9
    spiked[cutoff + spike_offset] = -1e9
    new_values, new_staleness = align_covariate(
        grid, _series(list(spiked), list(spiked.values())), lag, max_staleness
    )

    np.testing.assert_array_equal(new_values[: step + 1], base_values[: step + 1])
    np.testing.assert_array_equal(new_staleness[: step + 1], base_staleness[: step + 1])
