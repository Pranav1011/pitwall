"""Figures: degradation curves, pit-window sensitivity, predicted-vs-actual.
Saves PNGs to figures/. Every figure is built from measured artifacts.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pitwall.artifacts import StaleModelError, load_fitted_model
from pitwall.config import (
    CLEAN_PARQUET,
    DRY_COMPOUNDS,
    FIGURES_DIR,
    RACE_LAPS_PARQUET,
    RESULTS_DIR,
)
from pitwall.model import predict_main, prepare
from pitwall.race_laps import load_race_laps
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
    """One-stop pit-lap sweep for one race: expected race time relative to the best
    one-stop lap, with 95% Monte-Carlo error bars, plus the P5-P95 outcome band.
    Pit laps with no feasible compound plan are marked, not interpolated across."""
    from pitwall.simulate import enumerate_pit_plans, simulate_plan
    resid = residual_sigma(model, clean)
    caps = stint_caps(clean)
    circuit, season, n_sims = "Bahrain Grand Prix", 2023, 2000
    total = int(raw[raw["race"] == circuit]["LapNumber"].max())
    env = build_env(circuit, season, total, model, clean, raw, resid, caps)
    rng = np.random.default_rng(0)
    laps, sims = [], []
    for pl in enumerate_pit_plans(total):
        if len(pl) == 1:
            laps.append(pl[0])
            sims.append(simulate_plan(env, pl, n_sims, rng))
    x = np.array(laps)
    ok = np.array([s is not None for s in sims])

    def col(key):
        return np.array([s[key] if s else np.nan for s in sims])

    mean, se, p5, p95 = col("mean"), col("mc_se"), col("p5"), col("p95")
    best = int(np.nanargmin(mean))
    diff = mean - mean[best]
    diff_err = 1.96 * np.sqrt(se ** 2 + se[best] ** 2)
    diff_err[best] = 0.0  # the reference plan itself
    step = np.diff(x).min()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax in (a1, a2):
        for i, xi in enumerate(x[~ok]):
            ax.axvspan(xi - step / 2, xi + step / 2, color="#bbbbbb", alpha=0.35, lw=0,
                       label="infeasible: no legal two-compound plan" if i == 0 else None)
    a1.axhline(0, color="k", lw=0.6, alpha=0.4)
    a1.errorbar(x[ok], diff[ok], yerr=diff_err[ok], fmt="o", ms=4, capsize=2,
                color="#1f77b4", label="vs best lap (±95% Monte-Carlo error)")
    a1.plot(x[best], 0, "o", ms=9, mfc="none", mec="#d1122b", mew=1.8,
            label=f"lowest expected time: lap {x[best]}")
    a1.set_xlabel("One-stop pit lap")
    a1.set_ylabel("Expected race time minus best one-stop lap (s)")
    a1.set_title("Expected race time relative to the best one-stop lap")
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)

    for seg in np.split(np.arange(len(x)), np.flatnonzero(~ok)):
        seg = seg[ok[seg]]
        if len(seg):
            a2.fill_between(x[seg], p5[seg], p95[seg], alpha=0.25, color="#1f77b4", lw=0)
            a2.plot(x[seg], mean[seg], "-o", ms=3, color="#1f77b4")
    a2.fill_between([], [], [], alpha=0.25, color="#1f77b4", label="P5–P95 outcomes")
    a2.plot([], [], "-o", ms=3, color="#1f77b4", label="expected")
    a2.set_xlabel("One-stop pit lap")
    a2.set_ylabel("Simulated race time (s)")
    a2.set_title("Spread of simulated outcomes per pit lap")
    a2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.suptitle(f"One-stop pit-lap sweep — {season} {circuit} ({total} laps, "
                 f"{n_sims:,} simulations per lap)")
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
    try:
        model = load_fitted_model(RESULTS_DIR / "model_full.pkl", CLEAN_PARQUET)
    except StaleModelError as err:
        raise SystemExit(f"error: {err}") from None
    raw = load_race_laps(RACE_LAPS_PARQUET)
    fig_degradation(clean, model)
    fig_pit_window(clean, raw, model)
    fig_pred_actual()
    print("figures written to figures/")


if __name__ == "__main__":
    main()
