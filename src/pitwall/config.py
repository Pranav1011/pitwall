"""Shared constants for PitWall."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

CLEAN_PARQUET = DATA_DIR / "laps_clean.parquet"

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]

# stint identity: a tyre run is unique within a race + driver + stint number
STINT_KEYS = ["season", "round", "Driver", "Stint"]
RACE_KEYS = ["season", "round"]

# Tier 1 thresholds
OUTLIER_PCT = 0.07  # drop laps > median * (1 + this) within a stint
MIN_STINT_LAPS = 5
