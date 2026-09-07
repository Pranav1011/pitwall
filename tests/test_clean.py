"""One test per exclusion rule, plus a regression pin on the clean-lap count."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitwall import clean
from pitwall.config import RESULTS_DIR


def _base(n=10, **over):
    df = pd.DataFrame({
        "LapTime": np.full(n, 90.0),
        "TyreLife": np.arange(1, n + 1, dtype=float),
        "LapNumber": np.arange(2, n + 2, dtype=float),
        "PitInTime": np.nan, "PitOutTime": np.nan,
        "TrackStatus": ["1"] * n,
        "Compound": ["MEDIUM"] * n,
        "season": 2023, "round": 1, "Driver": "VER", "Stint": 1.0,
    })
    for k, v in over.items():
        df[k] = v
    return df


def test_null_laptime():
    df = _base()
    df.loc[0, "LapTime"] = np.nan
    assert clean.rule_null_laptime(df).sum() == 1


def test_null_tyrelife():
    df = _base()
    df.loc[2, "TyreLife"] = np.nan
    assert clean.rule_null_tyrelife(df).sum() == 1


def test_lap_one():
    df = _base()
    df.loc[0, "LapNumber"] = 1
    assert clean.rule_lap_one(df).sum() == 1


def test_pit_laps():
    df = _base()
    df.loc[0, "PitInTime"] = 12.3
    df.loc[1, "PitOutTime"] = 4.5
    assert clean.rule_pit_laps(df).sum() == 2


def test_non_green():
    df = _base()
    df.loc[0, "TrackStatus"] = "4"
    df.loc[1, "TrackStatus"] = "12"
    assert clean.rule_non_green(df).sum() == 2


def test_wet_inter_laps():
    df = _base()
    df.loc[0, "Compound"] = "WET"
    df.loc[1, "Compound"] = "INTERMEDIATE"
    assert clean.rule_wet_inter_laps(df).sum() == 2


def test_mixed_condition_race():
    df = _base(n=6)
    df.loc[0, "Compound"] = "INTERMEDIATE"  # one wet lap taints the whole race
    assert clean.rule_mixed_condition_race(df).sum() == 6


def test_unknown_compound():
    df = _base()
    df.loc[0, "Compound"] = None
    assert clean.rule_unknown_compound(df).sum() == 1


def test_stint_outlier():
    df = _base()
    df.loc[5, "LapTime"] = 90.0 * 1.2  # 20% over median
    assert clean.rule_stint_outlier(df).iloc[5]


def test_short_stint():
    df = _base(n=4)  # whole stint has < 5 laps
    assert clean.rule_short_stint(df).all()


def test_clean_pipeline_runs():
    df = _base(n=12)
    out, table = clean.clean_laps(df)
    assert len(out) == 12  # a clean stint survives every rule
    assert list(table["rule"]) == [r[0] for r in clean.RULES]


def test_clean_count_regression():
    """Pin the clean-lap count so a silent change to any rule is caught. The pin is
    written by scripts/build_clean.py and committed alongside the data."""
    summ = RESULTS_DIR / "tier1_summary.json"
    pin = RESULTS_DIR / "clean_count.pin"
    if not summ.exists() or not pin.exists():
        pytest.skip("run scripts/build_clean.py first")
    s = json.loads(Path(summ).read_text())
    assert s["valid_laps"] == int(pin.read_text().strip())
