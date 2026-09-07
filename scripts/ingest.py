"""Background ingestion: pull FastF1 race sessions for 2022-2024, one parquet per race.

Network-bound and the biggest time risk, so: cache is enabled, every race is wrapped
in try/except (failures are logged and skipped, never fatal), and a manifest records
exactly which races landed so downstream tiers report a real season/race count.

Run:  python scripts/ingest.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import fastf1
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "cache"
MANIFEST = ROOT / "data" / "ingest_manifest.json"
SEASONS = [2022, 2023, 2024]

# Per-lap columns we keep. Names follow FastF1's laps schema.
KEEP = [
    "Driver", "DriverNumber", "Team", "LapNumber", "LapTime", "Stint",
    "Compound", "TyreLife", "FreshTyre", "TrackStatus", "Position",
    "PitInTime", "PitOutTime", "IsAccurate",
]


def load_race(year: int, rnd: int, event_name: str) -> pd.DataFrame | None:
    ses = fastf1.get_session(year, rnd, "R")
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = ses.laps
    if laps is None or len(laps) == 0:
        return None
    df = laps.copy()
    for c in KEEP:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[KEEP].copy()
    # timedeltas -> seconds so parquet + modelling are numeric
    for c in ["LapTime", "PitInTime", "PitOutTime"]:
        df[c] = pd.to_timedelta(df[c]).dt.total_seconds()
    df["season"] = year
    df["round"] = rnd
    df["race"] = event_name
    return df


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))

    manifest: list[dict] = []
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    done = {(m["season"], m["round"]) for m in manifest if m["status"] == "ok"}

    for year in SEASONS:
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"[schedule fail] {year}: {e}", flush=True)
            continue
        for _, ev in sched.iterrows():
            rnd = int(ev["RoundNumber"])
            name = str(ev["EventName"])
            if rnd == 0 or (year, rnd) in done:
                continue
            slug = f"{year}_{rnd:02d}_{name.replace(' ', '_')}"
            out = RAW / f"{slug}.parquet"
            t0 = time.time()
            try:
                df = load_race(year, rnd, name)
                if df is None:
                    raise ValueError("no laps")
                df.to_parquet(out, index=False)
                rec = {"season": year, "round": rnd, "race": name,
                       "status": "ok", "laps": len(df), "secs": round(time.time() - t0, 1)}
                print(f"[ok] {slug}: {len(df)} laps ({rec['secs']}s)", flush=True)
            except Exception as e:
                rec = {"season": year, "round": rnd, "race": name,
                       "status": "fail", "error": f"{type(e).__name__}: {e}"}
                print(f"[fail] {slug}: {e}", flush=True)
            manifest = [m for m in manifest
                        if not (m["season"] == year and m["round"] == rnd)]
            manifest.append(rec)
            MANIFEST.write_text(json.dumps(manifest, indent=2))

    ok = sum(1 for m in manifest if m["status"] == "ok")
    seasons = sorted({m["season"] for m in manifest if m["status"] == "ok"})
    print(f"\nINGEST DONE: {ok} races across seasons {seasons}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
