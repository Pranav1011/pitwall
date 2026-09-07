"""Tier 2 driver: grouped-by-race cross-validation of the degradation model vs the
per-compound OLS baseline. Writes results/model_metrics.json and results/model_full.pkl.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from pitwall.config import CLEAN_PARQUET, DRY_COMPOUNDS, RESULTS_DIR
from pitwall.model import (
    fit_baseline,
    fit_fair_baseline,
    fit_main,
    predict_baseline,
    predict_fair_baseline,
    predict_main,
    prepare,
)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def per_compound_mae(comp: np.ndarray, err: np.ndarray) -> dict:
    out = {}
    for c in DRY_COMPOUNDS:
        mask = comp == c
        if mask.any():
            out[c] = round(float(np.mean(np.abs(err[mask]))), 4)
    return out


def main() -> None:
    df = prepare(pd.read_parquet(CLEAN_PARQUET))
    df["race_id"] = df["season"].astype(str) + "_" + df["round"].astype(str)
    n_races = df["race_id"].nunique()

    # Leave-one-race-out, but only score races whose circuit has cross-season data
    # (so a circuit intercept is always estimable from the training set). Single-
    # occurrence circuits still train the compound slopes; they just aren't held out.
    circuit_races = df.groupby("circuit")["race_id"].nunique()
    repeat_circuits = set(circuit_races[circuit_races >= 2].index)
    eligible = [r for r in df["race_id"].unique()
                if df.loc[df.race_id == r, "circuit"].iloc[0] in repeat_circuits]
    print(f"{len(df)} laps, {n_races} races; LORO over {len(eligible)} races "
          f"at {len(repeat_circuits)} cross-season circuits")

    y_true, y_model, y_base, y_fair, comps = [], [], [], [], []
    methods = []
    for r in eligible:
        train = df[df.race_id != r]
        test = df[df.race_id == r]
        bmodel = fit_baseline(train)
        fmodel = fit_fair_baseline(train)
        mmodel = fit_main(train)
        methods.append(mmodel.method)
        y_true.append(test["LapTime"].to_numpy())
        y_base.append(predict_baseline(bmodel, test))
        y_fair.append(predict_fair_baseline(fmodel, test))
        y_model.append(predict_main(mmodel, test))
        comps.append(test["compound"].to_numpy())

    y_true = np.concatenate(y_true)
    y_model = np.concatenate(y_model)
    y_base = np.concatenate(y_base)
    y_fair = np.concatenate(y_fair)
    comps = np.concatenate(comps)

    model_mae = mae(y_true, y_model)
    base_mae = mae(y_true, y_base)
    fair_mae = mae(y_true, y_fair)
    pc_model = per_compound_mae(comps, y_true - y_model)
    pc_base = per_compound_mae(comps, y_true - y_base)
    pc_fair = per_compound_mae(comps, y_true - y_fair)
    worst = max(pc_model, key=pc_model.get)

    # final fit on all data for figures / coefficients
    full = fit_main(df)

    metrics = {
        "laps": len(df),
        "races": n_races,
        "cv_races_scored": len(eligible),
        "cv_scheme": "leave-one-race-out over cross-season circuits "
                     "(held-out race never in train; circuit intercept from other season)",
        "model_method": full.method,
        "model_converged": full.converged,
        "fold_methods": methods,
        "model_cv_mae_s": round(model_mae, 4),
        "fair_baseline_cv_mae_s": round(fair_mae, 4),
        "fair_baseline_desc": "per-(circuit,compound) OLS of lap time on tyre_life",
        "improvement_vs_fair_pct": round(100 * (fair_mae - model_mae) / fair_mae, 1),
        "naive_baseline_cv_mae_s": round(base_mae, 4),
        "naive_baseline_desc": "per-compound OLS, no circuit (reference only)",
        "model_mae_by_compound": pc_model,
        "fair_baseline_mae_by_compound": pc_fair,
        "naive_baseline_mae_by_compound": pc_base,
        "model_worst_compound": worst,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(RESULTS_DIR / "model_full.pkl", "wb") as f:
        pickle.dump(full, f)
    # out-of-sample predictions for the predicted-vs-actual figure
    np.savez(RESULTS_DIR / "cv_predictions.npz",
             y_true=y_true, y_model=y_model, comps=comps)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
