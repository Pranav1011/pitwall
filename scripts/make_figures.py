"""Tier 4 figures: degradation curves, pit-window sensitivity, predicted-vs-actual.
Saves PNGs to figures/. Every figure is built from measured artifacts.
"""

from __future__ import annotations

import glob
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitwall.config import CLEAN_PARQUET, DRY_COMPOUNDS, FIGURES_DIR, RAW_DIR, RESULTS_DIR
from pitwall.model import predict_main, prepare
from pitwall.simulate import build_env, residual_sigma, stint_caps

COLORS = {"SOFT": "#d1122b", "MEDIUM": "#e8b000", "HARD": "#8a8a8a"}


def fig_degradation(clean, model):
    """Degradation isolated from fuel/circuit: for every lap, the tyre contribution is
    (lap time - model's fresh-tyre prediction for that same lap). Observed means of that
    quantity are overlaid on the model's degradation curve so both are on one scale."""
    d = prepare(clean)
    fresh = d.copy()
    fresh["tyre_life"] = 1.0
    fresh["tyre_life_sq"] = 1.0
    d["tyre_effect"] = d["LapTime"].to_numpy() - predict_main(model, fresh)
    caps = stint_caps(clean)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in DRY_COMPOUNDS:
        sub = d[(d["compound"] == c) & (d["tyre_life"] <= caps[c])]
        grp = sub.groupby(sub["tyre_life"].round())["tyre_effect"]
        life = grp.mean().index.values
        obs, se = grp.mean().values, (grp.std() / np.sqrt(grp.count())).values
        ax.fill_between(life, obs - se, obs + se, color=COLORS[c], alpha=0.18)
        ax.plot(life, obs, "o", ms=3, color=COLORS[c], alpha=0.55)
        gl = np.arange(1, caps[c] + 1)
        grid = pd.DataFrame({"compound": c, "tyre_life": gl.astype(float),
                             "tyre_life_sq": gl.astype(float) ** 2, "race_progress": 0.5,
                             "season": d["season"].max(), "circuit": d["circuit"].iloc[0],
                             "driver_season": "SIM"})
        fresh_g = grid.copy()
        fresh_g["tyre_life"] = 1.0
        fresh_g["tyre_life_sq"] = 1.0
        curve = predict_main(model, grid) - predict_main(model, fresh_g)
        ax.plot(gl, curve, "-", lw=2.2, color=COLORS[c], label=f"{c} (model)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("Tyre life (laps)")
    ax.set_ylabel("Tyre-wear lap-time loss vs fresh (s)")
    ax.set_title("Tyre degradation by compound — model vs fuel-corrected observed means (±SE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "degradation_by_compound.png", dpi=130)
    plt.close(fig)


def fig_pit_window(clean, raw, model):
    """Expected race time vs 1-stop pit lap for one race, with P5-P95 band."""
    from pitwall.simulate import enumerate_pit_plans, simulate_plan
    resid = residual_sigma(model, clean)
    caps = stint_caps(clean)
    circuit = "Bahrain Grand Prix"
    total = int(raw[raw["race"] == circuit]["LapNumber"].max())
    env = build_env(circuit, 2023, total, model, clean, raw, resid, caps)
    rng = np.random.default_rng(0)
    ones = []
    for pl in enumerate_pit_plans(total):
        if len(pl) != 1:
            continue
        s = simulate_plan(env, pl, 2000, rng)
        if s:
            ones.append((pl[0], s["mean"], s["p5"], s["p95"]))
    ones.sort()
    x = np.array([o[0] for o in ones])
    mean = np.array([o[1] for o in ones])
    p5 = np.array([o[2] for o in ones])
    p95 = np.array([o[3] for o in ones])
    opt = int(x[int(np.argmin(mean))])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    # left: expected time, zoomed to show the (shallow) optimum
    a1.plot(x, mean, "-o", ms=4, color="#1f77b4")
    a1.axvline(opt, ls="--", color="#d1122b", label=f"pace optimum: lap {opt}")
    a1.set_ylim(mean.min() - 1.5, mean.min() + 8)
    a1.set_xlabel("One-stop pit lap")
    a1.set_ylabel("Expected race time (s)")
    a1.set_title("Expected time is shallow (~few s):\nreal timing is set by the undercut")
    a1.legend()
    # right: full P5-P95 outcome spread (safety-car driven)
    a2.fill_between(x, p5, p95, alpha=0.2, color="#1f77b4", label="P5–P95 outcomes")
    a2.plot(x, mean, "-o", ms=4, color="#1f77b4", label="expected")
    a2.set_xlabel("One-stop pit lap")
    a2.set_ylabel("Simulated race time (s)")
    a2.set_title("Outcome spread is dominated by\nsafety-car risk (±~75 s)")
    a2.legend()
    fig.suptitle(f"Pit-window sensitivity — {circuit} ({total} laps, 1-stop)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pit_window_sensitivity.png", dpi=130)
    plt.close(fig)


def fig_pred_actual():
    """Out-of-sample predicted vs actual lap time (from CV)."""
    npz = RESULTS_DIR / "cv_predictions.npz"
    if not npz.exists():
        return
    d = np.load(npz, allow_pickle=True)
    yt, ym = d["y_true"], d["y_model"]
    idx = np.random.default_rng(0).choice(len(yt), size=min(4000, len(yt)), replace=False)
    mae = float(np.mean(np.abs(yt - ym)))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt[idx], ym[idx], s=4, alpha=0.25, color="#1f77b4")
    lo, hi = np.percentile(yt, 1), np.percentile(yt, 99)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual lap time (s)")
    ax.set_ylabel("Predicted lap time (s)")
    ax.set_title(f"Out-of-sample predicted vs actual (LORO)\nMAE = {mae:.2f} s/lap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    clean = pd.read_parquet(CLEAN_PARQUET)
    with open(RESULTS_DIR / "model_full.pkl", "rb") as f:
        model = pickle.load(f)
    raw = pd.concat((pd.read_parquet(f) for f in glob.glob(str(RAW_DIR / "*.parquet"))),
                    ignore_index=True)
    fig_degradation(clean, model)
    fig_pit_window(clean, raw, model)
    fig_pred_actual()
    print("figures written to figures/")


if __name__ == "__main__":
    main()
