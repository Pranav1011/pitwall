"""Tier 3 — Monte-Carlo pit-strategy simulator.

Race time for a stint plan decomposes into a deterministic part (sum of the
degradation model's predicted lap times) plus stochastic parts drawn per simulation:

  * lap-time noise   — sum of N(0, sigma) residuals over the race = N(0, sigma*sqrt(L)),
                       sigma from the fitted model's residual spread.
  * pit-lane loss    — sampled per stop from a per-circuit empirical pool built from
                       real in/out-lap deltas (not a hardcoded guess).
  * safety car       — Bernoulli occurrence at the per-circuit historical rate, lap
                       drawn from the circuit's historical SC-lap pool; SC laps cost
                       extra time, and a stop made in the SC window is discounted
                       (the "cheap stop"), which is why SC timing changes strategy.

All per-circuit rates are estimated from the ingested data. The SC model is a
deliberate simplification (fixed duration and slowdown factor) — stated in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitwall.config import DRY_COMPOUNDS, STINT_KEYS
from pitwall.model import MainModel, predict_main

# SC model constants (documented simplifications)
SC_WINDOW = 3          # laps: a stop within this of the SC lap is "cheap"
SC_PIT_DISCOUNT = 0.5  # fraction of pit loss saved by stopping under SC
SC_DURATION = 3        # laps run behind the safety car
SC_SLOWDOWN = 0.40     # SC laps are ~40% slower than green
MIN_STINT = 8          # laps: shortest plausible stint
GRID_STEP = 2          # pit-lap grid resolution


@dataclass
class RaceEnv:
    circuit: str
    season: int
    total_laps: int
    green_ref: float          # median green lap time (s)
    sc_prob: float            # P(>=1 safety car)
    sc_laps: np.ndarray       # historical SC start laps at this circuit
    pit_loss: np.ndarray      # empirical pit-loss pool (s)
    resid_sigma: float        # model residual std (s/lap)
    caps: dict                # compound -> max usable stint laps (physical tyre life)
    model: MainModel


def stint_caps(clean: pd.DataFrame, pct: float = 0.90) -> dict:
    """Max usable stint length per compound = a high percentile of stints actually run.
    Bounds the model to its supported range so it can't 'run' an impossible tyre life."""
    sizes = clean.groupby(STINT_KEYS + ["Compound"]).size().reset_index(name="laps")
    caps = {}
    for c in DRY_COMPOUNDS:
        s = sizes[sizes["Compound"] == c]["laps"]
        caps[c] = int(s.quantile(pct)) if len(s) else 30
    return caps


# ---------- estimating the stochastic inputs from data ----------

def residual_sigma(model: MainModel, clean: pd.DataFrame) -> float:
    from pitwall.model import prepare
    d = prepare(clean)
    d["driver_season"] = "SIM"  # RE=0: population-level residual spread
    pred = predict_main(model, d)
    return float(np.std(d["LapTime"].to_numpy() - pred))


def pit_loss_pool(raw: pd.DataFrame, circuit: str, green_ref: float) -> np.ndarray:
    """Time lost pitting = (in-lap excess + out-lap excess) over a green lap."""
    r = raw[raw["race"] == circuit]
    losses = []
    for (_, _, _), g in r.groupby(["season", "round", "Driver"]):
        g = g.sort_values("LapNumber")
        for _, lap in g[g["PitInTime"].notna()].iterrows():
            inlap = lap["LapTime"]
            out = g[g["LapNumber"] == lap["LapNumber"] + 1]
            if pd.isna(inlap) or out.empty or pd.isna(out["LapTime"].iloc[0]):
                continue
            loss = (inlap - green_ref) + (out["LapTime"].iloc[0] - green_ref)
            if 5 < loss < 60:  # guard against SC/red-flag in/out laps
                losses.append(loss)
    return np.array(losses) if losses else np.array([green_ref * 0.25])


def sc_stats(raw: pd.DataFrame, circuit: str) -> tuple[float, np.ndarray]:
    """P(safety car) and historical SC start laps for a circuit."""
    r = raw[raw["race"] == circuit]
    races = r.groupby(["season", "round"])
    had_sc, sc_laps = [], []
    for _, g in races:
        sc = g[g["TrackStatus"].astype(str).str.contains("4")]
        had_sc.append(len(sc) > 0)
        if len(sc):
            sc_laps.append(int(sc["LapNumber"].min()))
    prob = float(np.mean(had_sc)) if had_sc else 0.0
    return prob, np.array(sc_laps) if sc_laps else np.array([1])


def build_env(circuit: str, season: int, total_laps: int, model: MainModel,
              clean: pd.DataFrame, raw: pd.DataFrame, resid_sigma: float,
              caps: dict) -> RaceEnv:
    green = float(clean[clean["race"] == circuit]["LapTime"].median())
    if not np.isfinite(green):
        green = float(clean["LapTime"].median())
    prob, sc_laps = sc_stats(raw, circuit)
    return RaceEnv(circuit, season, total_laps, green, prob, sc_laps,
                   pit_loss_pool(raw, circuit, green), resid_sigma, caps, model)


# ---------- deterministic degradation cost ----------

def stint_laptimes(env: RaceEnv, compound: str, start_lap: int, length: int) -> np.ndarray:
    laps = np.arange(length)
    df = pd.DataFrame({
        "compound": compound,
        "tyre_life": laps + 1.0,
        "tyre_life_sq": (laps + 1.0) ** 2,
        "race_progress": (start_lap + laps) / env.total_laps,
        "season": env.season,
        "circuit": env.circuit,
        "driver_season": "SIM",
    })
    return predict_main(env.model, df)


def plan_from_pitlaps(env: RaceEnv, pit_laps: list[int]) -> list[tuple[str, int, int]] | None:
    """Turn pit laps into stints and greedily assign each stint its fastest compound
    whose tyre life covers the stint, then enforce the >=2-distinct-compound rule.
    Returns None if any stint is too long for every compound (infeasible plan)."""
    bounds = [0, *pit_laps, env.total_laps]
    stints = [(bounds[i] + 1, bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]
    chosen, costs = [], []
    for start, length in stints:
        feasible = {c: float(stint_laptimes(env, c, start, length).sum())
                    for c in DRY_COMPOUNDS if env.caps[c] >= length}
        if not feasible:  # no tyre can physically last this stint
            return None
        chosen.append(min(feasible, key=feasible.get))
        costs.append(feasible)
    if len(set(chosen)) < 2:  # all-same -> swap a stint to a distinct legal compound
        options = [(i, sorted(costs[i].values())[1] - min(costs[i].values()))
                   for i in range(len(chosen)) if len(costs[i]) > 1]
        if not options:
            return None
        swap_i = min(options, key=lambda t: t[1])[0]
        alt = min((c for c in costs[swap_i] if c != chosen[swap_i]),
                  key=lambda c: costs[swap_i][c])
        chosen[swap_i] = alt
    return [(chosen[i], stints[i][0], stints[i][1]) for i in range(len(stints))]


def deterministic_time(env: RaceEnv, plan: list[tuple[str, int, int]]) -> float:
    return float(sum(stint_laptimes(env, c, s, ln).sum() for c, s, ln in plan))


# ---------- Monte-Carlo over a plan ----------

def simulate_plan(env: RaceEnv, pit_laps: list[int], n: int,
                  rng: np.random.Generator) -> dict | None:
    plan = plan_from_pitlaps(env, pit_laps)
    if plan is None:
        return None
    base = deterministic_time(env, plan)
    n_stops = len(pit_laps)

    lap_noise = rng.normal(0, env.resid_sigma * np.sqrt(env.total_laps), n)

    pit_total = np.zeros(n)
    sc_occur = rng.random(n) < env.sc_prob
    sc_lap = rng.choice(env.sc_laps, size=n)
    for p in pit_laps:
        loss = rng.choice(env.pit_loss, size=n)
        cheap = sc_occur & (np.abs(sc_lap - p) <= SC_WINDOW)
        loss = np.where(cheap, loss * (1 - SC_PIT_DISCOUNT), loss)
        pit_total += loss

    sc_penalty = sc_occur * SC_DURATION * SC_SLOWDOWN * env.green_ref
    times = base + lap_noise + pit_total + sc_penalty
    return {
        "pit_laps": pit_laps,
        "n_stops": n_stops,
        "compounds": [c for c, _, _ in plan],
        "mean": float(times.mean()),
        "p5": float(np.percentile(times, 5)),
        "p95": float(np.percentile(times, 95)),
    }


def enumerate_pit_plans(total_laps: int) -> list[list[int]]:
    grid = list(range(MIN_STINT, total_laps - MIN_STINT + 1, GRID_STEP))
    plans = [[p] for p in grid]  # one-stop
    for i, p1 in enumerate(grid):  # two-stop
        for p2 in grid[i + 1:]:
            if p2 - p1 >= MIN_STINT and total_laps - p2 >= MIN_STINT:
                plans.append([p1, p2])
    return plans


def best_plan(env: RaceEnv, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    results = [simulate_plan(env, pl, n, rng)
               for pl in enumerate_pit_plans(env.total_laps)]
    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r["mean"])
    return {"best": results[0], "all": results}
