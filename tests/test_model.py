"""Model behaviour tests: degradation monotonicity and baseline sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pitwall.model import fit_main, predict_main, prepare


def _synth_race(circuit, season, rnd, base, slope, n_laps=40, driver="AAA"):
    """One driver, one long green stint on MEDIUM with linear degradation."""
    life = np.arange(1, n_laps + 1)
    return pd.DataFrame({
        "LapTime": base + slope * life + np.random.default_rng(rnd).normal(0, 0.05, n_laps),
        "TyreLife": life.astype(float), "LapNumber": life.astype(float),
        "Compound": "MEDIUM", "season": season, "round": rnd,
        "Driver": driver, "race": circuit,
    })


def _dataset():
    # two circuits, each in two seasons, so circuit intercepts are estimable
    parts = [
        _synth_race("CircuitA", 2022, 1, base=90, slope=0.10),
        _synth_race("CircuitA", 2023, 2, base=91, slope=0.10),
        _synth_race("CircuitB", 2022, 3, base=78, slope=0.08),
        _synth_race("CircuitB", 2023, 4, base=79, slope=0.08),
    ]
    return prepare(pd.concat(parts, ignore_index=True))


def test_predicted_laptime_increases_with_tyre_life():
    """Within a compound at a fixed circuit, predicted lap time must rise as the
    tyre ages (positive degradation)."""
    m = fit_main(_dataset())
    grid = pd.DataFrame({
        "compound": "MEDIUM", "tyre_life": np.arange(1, 30, dtype=float),
        "tyre_life_sq": np.arange(1, 30, dtype=float) ** 2,
        "race_progress": 0.5, "season": 2023, "circuit": "CircuitA",
        "driver_season": "SIM",
    })
    pred = predict_main(m, grid)
    diffs = np.diff(pred)
    assert (diffs > 0).mean() > 0.9  # monotone increasing over the stint


def test_model_recovers_circuit_offset():
    """Circuit centering should place CircuitB predictions well below CircuitA."""
    m = fit_main(_dataset())
    g = pd.DataFrame({
        "compound": ["MEDIUM", "MEDIUM"], "tyre_life": [10.0, 10.0],
        "tyre_life_sq": [100.0, 100.0], "race_progress": [0.5, 0.5],
        "season": [2023, 2023], "circuit": ["CircuitA", "CircuitB"],
        "driver_season": ["SIM", "SIM"],
    })
    a, b = predict_main(m, g)
    assert a - b > 8  # ~12s base gap between the two circuits
