"""Build data/race_laps.parquet: the raw-lap columns the backtest and figures need.

Run after scripts/ingest.py. The output is committed so that Tier 3 and the figures
reproduce on a fresh clone without re-ingesting from FastF1.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from pitwall.config import RACE_LAPS_PARQUET, RAW_DIR
from pitwall.race_laps import derive_race_laps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                    help="directory of per-race parquets written by scripts/ingest.py")
    args = ap.parse_args()
    files = sorted(glob.glob(str(args.raw_dir / "*.parquet")))
    if not files:
        raise SystemExit(f"no ingested races in {args.raw_dir}; run scripts/ingest.py first")
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    out = derive_race_laps(raw)
    out.to_parquet(RACE_LAPS_PARQUET, index=False)
    n_races = out[["season", "round"]].drop_duplicates().shape[0]
    print(f"wrote {RACE_LAPS_PARQUET.name}: {len(out)} laps from {n_races} races")


if __name__ == "__main__":
    main()
