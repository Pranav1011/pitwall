"""Tier 3 driver: backtest the simulator's chosen strategy against what the field
actually did, per race. Agreement = same stop count AND every pit lap within +/-3 laps
of the field's median. Writes results/backtest.json and results/backtest_table.md.
"""

from __future__ import annotations

import glob
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from pitwall.config import CLEAN_PARQUET, RAW_DIR, RESULTS_DIR
from pitwall.model import prepare
from pitwall.simulate import (
    build_env,
    enumerate_pit_plans,
    residual_sigma,
    simulate_plan,
    stint_caps,
)

N_SIMS = 2000
LAP_TOL = 3


def actual_strategy(raw_race: pd.DataFrame, total_laps: int):
    """Field-consensus strategy: modal stop count, median pit laps for that count."""
    finishers = []
    for drv, g in raw_race.groupby("Driver"):
        if g["LapNumber"].max() < 0.9 * total_laps:
            continue
        pits = sorted(int(x) for x in g.loc[g["PitInTime"].notna(), "LapNumber"]
                      if x < total_laps)
        finishers.append((drv, pits))
    if not finishers:
        return None
    counts = Counter(len(p) for _, p in finishers)
    modal = counts.most_common(1)[0][0]
    same = [p for _, p in finishers if len(p) == modal]
    if modal == 0 or not same:
        return modal, []
    med = [int(np.median([p[i] for p in same])) for i in range(modal)]
    return modal, med


def agrees(pred_laps, act_laps) -> bool:
    if len(pred_laps) != len(act_laps):
        return False
    return all(abs(a - b) <= LAP_TOL for a, b in zip(sorted(pred_laps), sorted(act_laps)))


def main() -> None:
    clean = pd.read_parquet(CLEAN_PARQUET)
    with open(RESULTS_DIR / "model_full.pkl", "rb") as f:
        model = pickle.load(f)
    resid = residual_sigma(model, clean)
    caps = stint_caps(clean)

    raw = pd.concat((pd.read_parquet(f) for f in glob.glob(str(RAW_DIR / "*.parquet"))),
                    ignore_index=True)
    prep = prepare(clean)
    prep["race_id"] = prep["season"].astype(str) + "_" + prep["round"].astype(str)

    rows, agree_flags, stopcount_flags, pit_offsets = [], [], [], []
    for race_id, g in prep.groupby("race_id"):
        season = int(g["season"].iloc[0])
        rnd = int(g["round"].iloc[0])
        circuit = g["circuit"].iloc[0]
        raw_race = raw[(raw["season"] == season) & (raw["round"] == rnd)]
        total_laps = int(raw_race["LapNumber"].max())
        act = actual_strategy(raw_race, total_laps)
        if act is None or act[0] == 0:
            continue
        act_count, act_laps = act

        env = build_env(circuit, season, total_laps, model, clean, raw, resid, caps)
        rng = np.random.default_rng(0)
        sims = [simulate_plan(env, pl, N_SIMS, rng) for pl in enumerate_pit_plans(total_laps)]
        sims = [s for s in sims if s is not None]
        if not sims:
            continue
        sims.sort(key=lambda r: r["mean"])
        best = sims[0]

        ok = agrees(best["pit_laps"], act_laps)
        agree_flags.append(ok)
        stopcount_flags.append(best["n_stops"] == act_count)
        if best["n_stops"] == act_count:  # signed pit-lap offset where stop count matches
            pit_offsets += [p - a for p, a in
                            zip(sorted(best["pit_laps"]), sorted(act_laps))]
        # time the field's actual plan would cost in the sim (for disagreements)
        act_sim = simulate_plan(env, act_laps, N_SIMS, rng) if act_laps else None
        delta = round(act_sim["mean"] - best["mean"], 1) if act_sim else None
        rows.append({
            "race": f"{season} {circuit}", "actual_stops": act_count,
            "actual_pit_laps": act_laps, "pred_stops": best["n_stops"],
            "pred_pit_laps": best["pit_laps"], "pred_compounds": best["compounds"],
            "agree": ok, "pred_time_gain_s": delta,
        })

    rate = round(100 * np.mean(agree_flags), 1) if agree_flags else 0.0
    stop_rate = round(100 * np.mean(stopcount_flags), 1) if stopcount_flags else 0.0
    out = {
        "races_backtested": len(rows),
        "n_sims_per_plan": N_SIMS,
        "agreement_definition": f"same stop count and every pit lap within +/-{LAP_TOL}",
        "agreement_rate_pct": rate,
        "stop_count_agreement_pct": stop_rate,
        "median_pit_lap_offset": int(np.median(pit_offsets)) if pit_offsets else None,
        "mean_pit_lap_offset": round(float(np.mean(pit_offsets)), 1) if pit_offsets else None,
        "offset_note": "sim pits later than the field (positive) — the unmodelled "
                       "undercut/track-position effect makes real teams stop earlier",
        "resid_sigma_s": round(resid, 3),
        "stint_caps": caps,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "backtest.json").write_text(json.dumps(out, indent=2))

    cols = ["race", "actual_stops", "actual_pit_laps", "pred_stops", "pred_pit_laps",
            "pred_compounds", "agree", "pred_time_gain_s"]
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (RESULTS_DIR / "backtest_table.md").write_text("\n".join(md))

    off = round(float(np.mean(pit_offsets)), 1) if pit_offsets else None
    print(f"races backtested: {len(rows)}   strict agreement: {rate}%   "
          f"stop-count agreement: {stop_rate}%   mean pit-lap offset: {off}   "
          f"resid sigma: {resid:.3f}s")
    for r in rows:
        print(f"  {'OK ' if r['agree'] else '   '} {r['race'][:28]:28} "
              f"act {r['actual_stops']}@{r['actual_pit_laps']}  "
              f"pred {r['pred_stops']}@{r['pred_pit_laps']}")


if __name__ == "__main__":
    main()
