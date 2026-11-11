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
                    "PitInTime", "TrackStatus"]


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

    Two kinds of tyre change are not strategy and are dropped: a change on lap 1 or 2
    (damage or a first-lap incident), and a change under a red flag, which is free
    because the race is suspended. Safety-car stops are kept: in many races the field's
    only planned stop was taken under the safety car."""
    pits = driver_laps[driver_laps["PitInTime"].notna()
                       & (driver_laps["LapNumber"] < total_laps)
                       & (driver_laps["LapNumber"] > FIRST_LAPS_NOT_STRATEGIC)
                       & ~driver_laps["TrackStatus"].astype(str).str.contains(RED_FLAG)]
    return sorted(int(x) for x in pits["LapNumber"])
