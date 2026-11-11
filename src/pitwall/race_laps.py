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
