"""The small slice of the raw FastF1 laps that Tier 3 and the figures need.

The simulator's per-circuit inputs (pit-lane loss, safety-car rate and laps) and the
backtest's "what the field actually did" both come from every raw lap, not only the
clean dry-tyre laps. Ingesting three seasons from FastF1 is slow and rate-limited, so
scripts/build_race_laps.py keeps just these columns in data/race_laps.parquet, which is
committed. That lets a fresh clone re-run the backtest and the figures without ingest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RACE_LAP_COLUMNS = ["season", "round", "race", "Driver", "LapNumber", "LapTime",
                    "PitInTime", "TrackStatus", "Compound", "TyreLife"]


def derive_race_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns Tier 3 reads, in a stable row order."""
    out = raw[RACE_LAP_COLUMNS].copy()
    out["TrackStatus"] = out["TrackStatus"].astype(str)
    return out.sort_values(["season", "round", "Driver", "LapNumber"],
                           ignore_index=True)


def load_race_laps(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"error: {path} is missing; build it from the raw ingest with "
                         "scripts/build_race_laps.py")
    return pd.read_parquet(path)


FIRST_LAPS_NOT_STRATEGIC = 2  # a tyre change on lap 1 or 2 is damage or a lap-1 incident
RED_FLAG = "5"  # FastF1 track-status code


def strategic_pit_laps(driver_laps: pd.DataFrame, total_laps: int) -> list[int]:
    """Laps on which one driver made a strategic pit stop.

    A pit entry counts only if the tyres changed: the next lap is on a different compound,
    or its tyre life does not simply continue (+1 lap) from the in-lap. This drops
    drive-throughs with no tyre change, such as the whole field being routed through
    the pit lane behind the safety car (2024 Qatar laps 36-37). If the next lap is
    missing, the entry is kept.

    Two kinds of tyre change are not strategy and are dropped as well: a change on lap 1
    or 2 (damage or a first-lap incident), and a change around a red flag, which is free
    because the race is suspended. FastF1 gives track status per lap, not per stop, so a
    red flag is detected approximately: the in-lap or the lap after it shows code 5.
    Safety-car stops are kept: in many races the field's only planned stop was taken
    under the safety car."""
    g = driver_laps.sort_values("LapNumber").reset_index(drop=True)
    nxt = g.shift(-1)
    consecutive = nxt["LapNumber"] == g["LapNumber"] + 1
    status = g["TrackStatus"].astype(str)
    next_status = nxt["TrackStatus"].where(consecutive).astype(str)
    red_flag = status.str.contains(RED_FLAG) | next_status.str.contains(RED_FLAG)
    same_tyres = (consecutive & (nxt["Compound"] == g["Compound"])
                  & (nxt["TyreLife"] == g["TyreLife"] + 1))
    pits = g[g["PitInTime"].notna()
             & (g["LapNumber"] < total_laps)
             & (g["LapNumber"] > FIRST_LAPS_NOT_STRATEGIC)
             & ~red_flag
             & ~same_tyres]
    return [int(x) for x in pits["LapNumber"]]
