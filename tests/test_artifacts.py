"""Fitted-model staleness guard: a model must match the clean data it is used with."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from pitwall.artifacts import StaleModelError, file_sha256, load_fitted_model
from pitwall.model import MainModel


def _write(tmp_path, stamp: bool):
    """Write a clean-data parquet and a model pickled against it (hash stamped or not)."""
    clean = tmp_path / "laps_clean.parquet"
    pd.DataFrame({"LapTime": [90.0, 91.0]}).to_parquet(clean)
    m = MainModel({}, 90.0, [2023], ["const"], np.zeros(1), {}, True, "MixedLM")
    if stamp:
        m.data_hash = file_sha256(clean)
    pkl = tmp_path / "model_full.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(m, f)
    return pkl, clean


def test_model_fitted_on_current_data_loads(tmp_path):
    pkl, clean = _write(tmp_path, stamp=True)
    assert load_fitted_model(pkl, clean).data_hash == file_sha256(clean)


def test_stale_model_is_refused(tmp_path):
    pkl, clean = _write(tmp_path, stamp=True)
    pd.DataFrame({"LapTime": [80.0, 81.0]}).to_parquet(clean)  # data rebuilt after fit
    with pytest.raises(StaleModelError, match="scripts/fit_model.py"):
        load_fitted_model(pkl, clean)


def test_model_without_hash_warns_but_loads(tmp_path):
    pkl, clean = _write(tmp_path, stamp=False)
    with pytest.warns(UserWarning, match="no data hash"):
        m = load_fitted_model(pkl, clean)
    assert m.method == "MixedLM"
