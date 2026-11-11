# Data

| File | What it is | Built by |
| --- | --- | --- |
| `laps_clean.parquet` | 53,854 clean dry-tyre laps (Tier 1) | `scripts/build_clean.py` |
| `race_laps.parquet` | 10 columns of every raw lap (74,601), for the backtest and figures | `scripts/build_race_laps.py` |
| `ingest_manifest.json` | which races were ingested | `scripts/ingest.py` |
| `raw/` | per-race FastF1 pulls; not committed | `scripts/ingest.py` |

## Source and terms

Both parquets are derived from Formula 1 live-timing data for the 2022–2024 seasons,
retrieved with [FastF1](https://docs.fastf1.dev/). Formula 1 holds the rights to that
data; this repository is not affiliated with Formula 1.

- The MIT licence in this repository covers the code only, not this data.
- The data is included only for non-commercial research and so that the results can be
  reproduced without re-downloading.
- It will be removed on request from the rights holder.
