"""Simulator sanity tests: the stop-count tradeoff must respond correctly to the
degradation-vs-pit-loss balance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitwall.model import fit_main, prepare
from pitwall.simulate import RaceEnv, best_plan


def _fit(slope: float):
    """Fit the real model on synthetic data with a chosen MEDIUM degradation slope."""
    def race(circ, season, rnd, base):
        life = np.arange(1, 41)
        return pd.DataFrame({
            "LapTime": base + slope * life, "TyreLife": life.astype(float),
            "LapNumber": life.astype(float), "Compound": "MEDIUM",
            "season": season, "round": rnd, "Driver": "AAA", "race": circ,
        })
    df = prepare(pd.concat([
        race("A", 2022, 1, 90), race("A", 2023, 2, 90),
        race("B", 2022, 3, 80), race("B", 2023, 4, 80),
    ], ignore_index=True))
    return fit_main(df)


def _env(model, pit_loss, caps=None, resid_sigma=0.0):
    return RaceEnv(
        circuit="A", season=2023, total_laps=40, green_ref=90.0,
        sc_prob=0.0, sc_laps=np.array([1]), pit_loss=np.array([pit_loss]),
        resid_sigma=resid_sigma, caps=caps or {"SOFT": 100, "MEDIUM": 100, "HARD": 100},
        model=model,
    )


def test_zero_deg_nonzero_pitloss_prefers_fewest_stops():
    """No tyre wear + real pit loss => stopping more only wastes pit time."""
    env = _env(_fit(slope=0.0), pit_loss=25.0)
    assert best_plan(env, n=300)["best"]["n_stops"] == 1


def test_deg_with_free_stops_prefers_more_stops():
    """Strong wear + free stops => fresher tyres win, so take the extra stop."""
    env = _env(_fit(slope=0.20), pit_loss=0.0)
    assert best_plan(env, n=300)["best"]["n_stops"] == 2


def test_percentile_band_is_ordered():
    env = _env(_fit(slope=0.05), pit_loss=22.0, resid_sigma=0.6)
    best = best_plan(env, n=500)["best"]
    assert best["p5"] < best["mean"] < best["p95"]


def test_monte_carlo_error_matches_lap_noise():
    """With only lap-time noise, the error of the mean is sigma*sqrt(laps)/sqrt(n)."""
    env = _env(_fit(slope=0.05), pit_loss=22.0, resid_sigma=0.6)
    best = best_plan(env, n=4000)["best"]
    expected = 0.6 * np.sqrt(40) / np.sqrt(4000)
    assert best["mc_se"] == pytest.approx(expected, rel=0.1)


def test_best_plan_raises_clear_error_when_no_plan_is_feasible():
    """Stint caps shorter than any legal stint leave no plan; say so, not IndexError."""
    env = _env(_fit(slope=0.05), pit_loss=22.0, caps={"SOFT": 5, "MEDIUM": 5, "HARD": 5})
    with pytest.raises(ValueError, match="no feasible pit plan"):
        best_plan(env, n=50)
