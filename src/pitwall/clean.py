"""Tier 1 — build a clean dry-tyre lap dataset.

Each exclusion rule is a named function returning a boolean mask of rows to REMOVE,
so every rule is independently testable and its effect is counted. Rules are applied
in order; the per-rule removal counts become the exclusion table in the README.
"""

from __future__ import annotations

import pandas as pd

from pitwall.config import (
    DRY_COMPOUNDS,
    MIN_STINT_LAPS,
    OUTLIER_PCT,
    RACE_KEYS,
    STINT_KEYS,
    WET_COMPOUNDS,
)


def rule_null_laptime(df: pd.DataFrame) -> pd.Series:
    """A lap with no recorded time is unusable."""
    return df["LapTime"].isna()


def rule_null_tyrelife(df: pd.DataFrame) -> pd.Series:
    """A lap with no tyre-life value can't enter a degradation model."""
    return df["TyreLife"].isna()


def rule_lap_one(df: pd.DataFrame) -> pd.Series:
    """Lap 1 is a standing start, not representative green-flag pace."""
    return df["LapNumber"] == 1


def rule_pit_laps(df: pd.DataFrame) -> pd.Series:
    """In-laps (PitInTime set) and out-laps (PitOutTime set) are pit-affected."""
    return df["PitInTime"].notna() | df["PitOutTime"].notna()


def rule_non_green(df: pd.DataFrame) -> pd.Series:
    """TrackStatus is a per-lap string of status codes; '1' throughout = all-green.
    Anything else means yellow(2)/SC(4)/red(5)/VSC(6,7) touched the lap."""
    return df["TrackStatus"].astype(str) != "1"


def rule_wet_inter_laps(df: pd.DataFrame) -> pd.Series:
    """Wet and intermediate laps — the model is dry-degradation only."""
    return df["Compound"].isin(WET_COMPOUNDS)


def rule_mixed_condition_race(df: pd.DataFrame) -> pd.Series:
    """Drop every remaining lap of any race that used a wet/inter compound at all:
    a dry<->wet compound switch means changing conditions the dry model can't own."""
    wet_races = (
        df.loc[df["Compound"].isin(WET_COMPOUNDS), RACE_KEYS]
        .drop_duplicates()
        .assign(_wet=True)
    )
    if wet_races.empty:
        return pd.Series(False, index=df.index)
    merged = df.merge(wet_races, on=RACE_KEYS, how="left")
    return merged["_wet"].notna().to_numpy()  # True (wet race) vs NaN (dry)


def rule_unknown_compound(df: pd.DataFrame) -> pd.Series:
    """Drop laps whose compound is missing or not a dry slick (can't model wear
    without knowing the tyre)."""
    return ~df["Compound"].isin(DRY_COMPOUNDS)


def rule_stint_outlier(df: pd.DataFrame) -> pd.Series:
    """Laps > OUTLIER_PCT slower than the driver's stint median: traffic, mistakes,
    lift-and-coast. Median is taken over the laps still valid at this point."""
    med = df.groupby(STINT_KEYS)["LapTime"].transform("median")
    return df["LapTime"] > med * (1 + OUTLIER_PCT)


def rule_short_stint(df: pd.DataFrame) -> pd.Series:
    """Stints with fewer than MIN_STINT_LAPS valid laps can't show a degradation trend."""
    n = df.groupby(STINT_KEYS)["LapTime"].transform("size")
    return n < MIN_STINT_LAPS


# applied in order; later rules see only rows that survived earlier ones.
# mixed_condition_race runs BEFORE wet laps are stripped so it can still see them.
RULES = [
    ("null_laptime", rule_null_laptime),
    ("null_tyrelife", rule_null_tyrelife),
    ("lap_1", rule_lap_one),
    ("pit_in_out_laps", rule_pit_laps),
    ("non_green_track_status", rule_non_green),
    ("mixed_condition_race", rule_mixed_condition_race),
    ("wet_intermediate_laps", rule_wet_inter_laps),
    ("unknown_compound", rule_unknown_compound),
    ("stint_median_outlier_7pct", rule_stint_outlier),
    ("short_stint_lt5", rule_short_stint),
]


def clean_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all rules in order. Returns (clean_df, exclusion_table)."""
    work = df.copy()
    rows = []
    for name, fn in RULES:
        before = len(work)
        remove = pd.Series(fn(work), index=work.index).fillna(False).astype(bool)
        work = work.loc[~remove].copy()
        rows.append({"rule": name, "removed": before - len(work), "remaining": len(work)})
    table = pd.DataFrame(rows)
    return work.reset_index(drop=True), table
