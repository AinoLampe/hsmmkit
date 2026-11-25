"""E10-T05: observation supports are validated against the emission family's support."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from regimekit.config.schema import DataSection
from regimekit.data.contracts import Support
from regimekit.data.prepare import Preparer
from regimekit.data.sources import load_data
from regimekit.utils.errors import DataError


def _prepare(values: Any, support: Support, *, names: list[str] | None = None, **kw: Any) -> Any:  # noqa: ANN401
    cols = names or [f"s{j}" for j in range(np.asarray(values).reshape(len(values), -1).shape[1])]
    frame = pd.DataFrame(np.asarray(values).reshape(len(values), -1), columns=cols)
    section = DataSection.model_validate(
        {"source": {"kind": "pandas", "observation_columns": cols, "dtype": "float64"}}
    )
    obs, covs = load_data(section, obj=frame)
    return Preparer(section, support=support, **kw).fit(obs, covs, len(values) - 1).transform(obs, covs)


def test_e10_t05_real_support_accepts_finite_values_and_masks_the_rest() -> None:
    """`real`: float64, non-finite values are masked, not rejected."""
    data = _prepare([0.5, np.nan, -1.0], "real")
    assert data.observations.dtype == np.float64
    assert np.asarray(data.mask)[:, 0].tolist() == [True, False, True]
    assert data.support == "real"


def test_e10_t05_count_support_accepts_integral_non_negative_values() -> None:
    """`count`: `int64` observations, float input accepted when integral; `support` is recorded."""
    data = _prepare([0.0, 3.0, 1.0, 7.0], "count")
    assert data.observations.dtype == np.int64
    assert np.asarray(data.observations)[:, 0].tolist() == [0, 3, 1, 7]
    assert data.support == "count"


@pytest.mark.parametrize(
    ("values", "row"), [([0.0, -1.0, 2.0], 1), ([1.0, 2.0, 2.5], 2)], ids=["negative", "fractional"]
)
def test_e10_t05_count_support_rejects_with_the_row_and_series(values: list[float], row: int) -> None:
    """`count` violated: DataError on `data.source.observation_columns`, naming the row and series."""
    with pytest.raises(DataError) as info:
        _prepare(values, "count", names=["visits"])
    assert info.value.path == "data.source.observation_columns"
    assert f"count support violated at row {row}, series visits" in str(info.value)


def test_e10_t05_binary_support_accepts_zero_one_and_rejects_others() -> None:
    """`binary`: values in {0, 1} become `int64`; any other value is a DataError."""
    data = _prepare([0.0, 1.0, 1.0, 0.0], "binary")
    assert data.observations.dtype == np.int64
    with pytest.raises(DataError, match="binary support violated at row 2, series s0"):
        _prepare([0.0, 1.0, 2.0], "binary")


def test_e10_t05_categorical_support_checks_the_range() -> None:
    """`categorical`: integers in `[0, n_categories)`, one series; `n_categories` is stored."""
    data = _prepare([0.0, 2.0, 1.0, 2.0], "categorical", n_categories=3)
    assert data.n_categories == 3
    assert data.support == "categorical"
    with pytest.raises(DataError, match="categorical support violated at row 1, series s0"):
        _prepare([0.0, 3.0, 1.0], "categorical", n_categories=3)
    with pytest.raises(DataError, match="single series"):
        _prepare([[0.0, 1.0], [1.0, 0.0]], "categorical", n_categories=2)


def test_e10_t05_categorical_without_n_categories_uses_max_plus_one_with_a_warning() -> None:
    """`n_categories=None`: `max + 1` and a warning."""
    with pytest.warns(UserWarning, match="max \\+ 1 = 3"):
        data = _prepare([0.0, 2.0, 1.0], "categorical")
    assert data.n_categories == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _prepare([0.0, 2.0, 1.0], "categorical", n_categories=3)  # no warning when given


def test_e10_t05_masked_values_are_not_checked_against_the_support() -> None:
    """A masked (missing) value does not violate a count support; it becomes 0 with `mask=False`."""
    data = _prepare([1.0, np.nan, 2.0], "count")
    assert np.asarray(data.mask)[:, 0].tolist() == [True, False, True]
    assert np.asarray(data.observations)[:, 0].tolist() == [1, 0, 2]
