"""Tier 3 inputs: the committed raw-lap slice and which tyre changes count as stops."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.race_laps import RACE_LAP_COLUMNS, derive_race_laps, strategic_pit_laps


def _driver(pit_laps: dict[int, str], total: int = 50,
            no_change: frozenset[int] = frozenset()) -> pd.DataFrame:
    """One driver's laps. pit_laps maps an in-lap number to that lap's TrackStatus; new
    tyres are fitted after each pit entry except those listed in no_change."""
    rows, life, compounds = [], 0, ["MEDIUM", "HARD"]
    stint = 0
    for n in range(1, total + 1):
        life += 1
        rows.append({"LapNumber": float(n), "TyreLife": float(life),
                     "Compound": compounds[stint % 2],
                     "PitInTime": 3600.0 if n in pit_laps else np.nan,
                     "TrackStatus": pit_laps.get(n, "1")})
        if n in pit_laps and n not in no_change:
            life, stint = 0, stint + 1
    return pd.DataFrame(rows)


def test_lap_one_and_two_tyre_changes_are_not_strategic():
    assert strategic_pit_laps(_driver({1: "1", 2: "1", 20: "1"}), 50) == [20]


def test_red_flag_tyre_change_is_not_strategic():
    assert strategic_pit_laps(_driver({15: "1", 30: "125"}), 50) == [15]


def test_tyre_change_on_the_lap_before_a_red_flag_is_not_strategic():
    laps = _driver({15: "1", 30: "1"})
    laps.loc[laps["LapNumber"] == 31, "TrackStatus"] = "5"
    assert strategic_pit_laps(laps, 50) == [15]


def test_safety_car_stop_is_kept():
    """Many races' only planned stop is taken under the safety car."""
    assert strategic_pit_laps(_driver({18: "4"}), 50) == [18]


def test_pit_lane_pass_without_a_tyre_change_is_not_a_stop():
    """2024 Qatar: the field pitted on lap 35, then was routed through the pit lane
    behind the safety car on laps 36 and 37 on the same tyres."""
    laps = _driver({35: "14", 36: "4", 37: "4"}, total=57, no_change=frozenset({36, 37}))
    assert strategic_pit_laps(laps, 57) == [35]


def test_final_lap_pit_entry_is_not_a_stop():
    assert strategic_pit_laps(_driver({25: "1", 50: "1"}), 50) == [25]


def test_derive_race_laps_keeps_only_tier3_columns():
    raw = _driver({20: "1"}).assign(season=2023, round=1, race="A", Driver="AAA",
                                     LapTime=90.0, FreshTyre=True)
    out = derive_race_laps(raw.iloc[::-1])
    assert list(out.columns) == RACE_LAP_COLUMNS
    assert out["LapNumber"].is_monotonic_increasing
