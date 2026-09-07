"""Tier 1 driver: load every ingested race, apply exclusion rules, write the clean set.

Writes data/laps_clean.parquet and results/exclusion_table.{csv,md}, and prints the
final valid-lap and stint counts.
"""

from __future__ import annotations

import glob
import json

import pandas as pd

from pitwall.clean import clean_laps
from pitwall.config import CLEAN_PARQUET, RAW_DIR, RESULTS_DIR, STINT_KEYS


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, sep, body])


def main() -> None:
    files = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    if not files:
        raise SystemExit("no ingested races yet in data/raw/")
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    seasons = sorted(raw["season"].unique().tolist())
    n_races = raw[["season", "round"]].drop_duplicates().shape[0]
    print(f"loaded {len(raw)} raw laps from {n_races} races, seasons {seasons}")

    clean, table = clean_laps(raw)
    n_stints = clean[STINT_KEYS].drop_duplicates().shape[0]

    RESULTS_DIR.mkdir(exist_ok=True)
    CLEAN_PARQUET.parent.mkdir(exist_ok=True)
    clean.to_parquet(CLEAN_PARQUET, index=False)
    table.to_csv(RESULTS_DIR / "exclusion_table.csv", index=False)
    (RESULTS_DIR / "exclusion_table.md").write_text(df_to_md(table))

    summary = {
        "seasons": seasons,
        "n_seasons": len(seasons),
        "races": n_races,
        "raw_laps": len(raw),
        "valid_laps": len(clean),
        "stints": n_stints,
    }
    (RESULTS_DIR / "tier1_summary.json").write_text(json.dumps(summary, indent=2))
    (RESULTS_DIR / "clean_count.pin").write_text(str(len(clean)))

    print("\n=== exclusion table ===")
    print(table.to_string(index=False))
    print(f"\nvalid laps: {len(clean)}   stints: {n_stints}")
    print(f"compounds kept: {clean['Compound'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
