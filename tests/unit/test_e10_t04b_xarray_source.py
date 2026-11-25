"""E10-T04b: the `xarray` source, the lazy optional import and the neutral snapshot hashes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regimekit.config.schema import DataSection
from regimekit.data.provenance import hash_arrays, hash_bytes, hash_file
from regimekit.data.sources import load_data
from regimekit.data.sources.base import get_source
from regimekit.plugins import REGISTRY
from regimekit.testing.source_contract import FIXTURES, check_source, xarray_case
from regimekit.utils.errors import ConfigError, DataError
from regimekit.utils.optional import optional_import

xr = pytest.importorskip("xarray")


def test_e10_t04b_xarray_source_passes_the_contract_suite() -> None:
    """AC1: the xarray source passes `source_contract` (with and without a time coordinate)."""
    get_source("csv")
    assert next(e for e in REGISTRY.entries("source") if e.name == "xarray").provenance == "regimekit.core"
    check_source("xarray", xarray_case())


def _dataset() -> object:
    frame = pd.read_csv(FIXTURES / "gauss.csv")
    return xr.Dataset(
        {c: ("time", frame[c].to_numpy()) for c in ("y1", "y2", "x")},
        coords={"time": pd.to_datetime(frame["t"]).to_numpy(), "seq": ("time", frame["seq"].to_numpy())},
    )


def test_e10_t04b_sequence_column_may_be_a_coordinate() -> None:
    """`sequence_column` names a coordinate; it fills `sequence_ids` and stays out of the values."""
    section = DataSection.model_validate(
        {
            "source": {
                "kind": "xarray",
                "observation_columns": ["y1", "y2"],
                "time_column": "time",
                "sequence_column": "seq",
            }
        }
    )
    obs, _ = load_data(section, obj=_dataset())
    assert obs.sequence_ids is not None
    assert sorted(set(obs.sequence_ids.tolist())) == [0, 1]
    assert obs.values.shape == (40, 2)


def test_e10_t04b_missing_xarray_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the library the source raises `ImportError("pip install hsmmkit[xarray]")`."""
    monkeypatch.setitem(sys.modules, "xarray", None)
    with pytest.raises(ImportError, match=r"pip install hsmmkit\[xarray\]"):
        optional_import("xarray", "xarray")
    section = DataSection.model_validate({"source": {"kind": "xarray", "observation_columns": ["y1"]}})
    with pytest.raises(ImportError, match=r"hsmmkit\[xarray\]"):
        load_data(section, obj=object())


def test_e10_t04b_xarray_path_errors_and_object_type(tmp_path: Path) -> None:
    """A missing file is a DataError on `data.source.path`; a foreign object a ConfigError on the kind."""
    missing = DataSection.model_validate(
        {"source": {"kind": "xarray", "observation_columns": ["y1"], "path": str(tmp_path / "x.nc")}}
    )
    with pytest.raises(DataError) as info:
        load_data(missing)
    assert info.value.path == "data.source.path"
    section = DataSection.model_validate({"source": {"kind": "xarray", "observation_columns": ["y1"]}})
    with pytest.raises(ConfigError) as info2:
        load_data(section, obj=pd.DataFrame({"y1": [1.0]}))
    assert info2.value.path == "data.source.kind"
    with pytest.raises(ConfigError) as info3:
        load_data(section)
    assert info3.value.path == "data.source.path"


def test_e10_t04b_provenance_hashes(tmp_path: Path) -> None:
    """The neutral hashes: bytes, file and array buffers (names, dtype and shape count)."""
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert hash_file(path) == hash_bytes(b"abc")
    a = np.arange(6.0)
    assert hash_arrays(["a"], [a]) == hash_arrays(["a"], [a.copy()])
    assert hash_arrays(["a"], [a]) != hash_arrays(["b"], [a])
    assert hash_arrays(["a"], [a]) != hash_arrays(["a"], [a.reshape(2, 3)])
    assert hash_arrays(["a"], [a]) != hash_arrays(["a"], [a.astype(np.float32)])


def test_e10_t04b_the_core_data_package_holds_no_cache_or_provider_code() -> None:
    """AC3: `regimekit/data` has no cache module and no finance or provider imports."""
    root = Path(__file__).resolve().parents[2] / "src" / "regimekit" / "data"
    assert not (root / "cache.py").exists()
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in ("regimekit.finance", "yfinance", "fredapi", "CachedProvider", "replay"):
            assert banned not in text, f"{path.name} mentions {banned}"
