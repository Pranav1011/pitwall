"""Tier 2 — tyre-degradation model.

Lap time is dominated by circuit and fuel, not tyre wear, so those are corrected
before degradation is estimated:

  * circuit  — absorbed by centering lap time on the TRAIN per-circuit mean (a circuit
               fixed intercept), which also makes cross-validation robust to circuits
               that appear in only some folds.
  * fuel/track evolution — race_progress fraction as a linear fixed effect.
  * tyre wear — compound-specific linear + quadratic tyre_life terms (the cliff is
               nonlinear), so each compound gets its own degradation curve.
  * driver  — random intercept grouped by driver-season (MixedLM).

Baseline for comparison: per-compound OLS of lap time on tyre_life (no circuit, no
fuel, no quadratic). If MixedLM fails to converge the main model falls back to OLS
with driver fixed effects, and `converged` records which path ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pitwall.config import DRY_COMPOUNDS, RACE_KEYS


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add modelling columns without dropping rows."""
    d = df.copy()
    d["tyre_life"] = d["TyreLife"].astype(float)
    d["tyre_life_sq"] = d["tyre_life"] ** 2
    maxlap = d.groupby(RACE_KEYS)["LapNumber"].transform("max")
    d["race_progress"] = d["LapNumber"] / maxlap
    d["driver_season"] = d["Driver"].astype(str) + "_" + d["season"].astype(str)
    d["circuit"] = d["race"].astype(str)
    d["compound"] = d["Compound"].astype(str)
    return d


# ---------- baseline: per-compound OLS of lap time on tyre_life ----------

def fit_baseline(train: pd.DataFrame) -> dict:
    models: dict[str, tuple[float, float]] = {}
    for c in DRY_COMPOUNDS:
        sub = train[train["compound"] == c]
        if len(sub) < 2:
            continue
        X = sm.add_constant(sub["tyre_life"].to_numpy())
        res = sm.OLS(sub["LapTime"].to_numpy(), X).fit()
        models[c] = (float(res.params[0]), float(res.params[1]))
    glob = float(train["LapTime"].mean())
    return {"per_compound": models, "global_mean": glob}


def predict_baseline(model: dict, test: pd.DataFrame) -> np.ndarray:
    out = np.full(len(test), model["global_mean"], dtype=float)
    tl = test["tyre_life"].to_numpy()
    comp = test["compound"].to_numpy()
    for i, c in enumerate(comp):
        if c in model["per_compound"]:
            b0, b1 = model["per_compound"][c]
            out[i] = b0 + b1 * tl[i]
    return out


# ---------- fair baseline: per-(circuit, compound) OLS of lap time on tyre_life ----------

def fit_fair_baseline(train: pd.DataFrame) -> dict:
    """A baseline that already knows the circuit: one OLS (lap time ~ tyre_life) per
    (circuit, compound). Improvement over this isolates fuel/nonlinearity/driver/pooling,
    not the trivial 'which track is this'."""
    cc: dict = {}
    for (circ, comp), sub in train.groupby(["circuit", "compound"]):
        if len(sub) < 5:
            continue
        try:
            X = sm.add_constant(sub["tyre_life"].to_numpy())
            res = sm.OLS(sub["LapTime"].to_numpy(), X).fit()
            b1 = float(res.params[1]) if len(res.params) > 1 else 0.0
            cc[(circ, comp)] = (float(res.params[0]), b1)
        except Exception:
            continue
    circ_mean = train.groupby("circuit")["LapTime"].mean().to_dict()
    return {"cc": cc, "circ_mean": circ_mean, "global": float(train["LapTime"].mean())}


def predict_fair_baseline(model: dict, test: pd.DataFrame) -> np.ndarray:
    circ = test["circuit"].to_numpy()
    comp = test["compound"].to_numpy()
    tl = test["tyre_life"].to_numpy()
    out = np.empty(len(test))
    for i in range(len(test)):
        key = (circ[i], comp[i])
        if key in model["cc"]:
            b0, b1 = model["cc"][key]
            out[i] = b0 + b1 * tl[i]
        else:  # circuit-aware fallback keeps the comparison fair
            out[i] = model["circ_mean"].get(circ[i], model["global"])
    return out


# ---------- main model: circuit-centered MixedLM (OLS fallback) ----------

_FE_COMPOUNDS = DRY_COMPOUNDS  # slope columns, one per compound


@dataclass
class MainModel:
    circuit_mean: dict
    global_mean: float
    seasons: list[int]
    columns: list[str]
    fe_params: np.ndarray
    re: dict  # driver_season -> intercept
    converged: bool
    method: str
    driver_fe: dict = field(default_factory=dict)  # OLS-fallback driver intercepts


def _design(df: pd.DataFrame, seasons: list[int]) -> tuple[np.ndarray, list[str]]:
    """Build the fixed-effects design matrix (circuit already centered out).
    `seasons` fixes the season-dummy columns so train and held-out data align."""
    n = len(df)
    cols: list[str] = ["const", "race_progress"]
    mat = [np.ones(n), df["race_progress"].to_numpy()]
    # season fixed effect (annual pace step); reference = first season in the list
    for s in sorted(seasons)[1:]:
        cols.append(f"season_{s}")
        mat.append((df["season"] == s).to_numpy(float))
    # compound main effects (drop first as reference)
    for c in _FE_COMPOUNDS[1:]:
        cols.append(f"comp_{c}")
        mat.append((df["compound"] == c).to_numpy(float))
    # compound-specific linear + quadratic tyre_life slopes
    tl = df["tyre_life"].to_numpy()
    tl2 = df["tyre_life_sq"].to_numpy()
    for c in _FE_COMPOUNDS:
        cols.append(f"tl_{c}")
        mat.append(tl * (df["compound"] == c).to_numpy(float))
    for c in _FE_COMPOUNDS:
        cols.append(f"tl2_{c}")
        mat.append(tl2 * (df["compound"] == c).to_numpy(float))
    return np.column_stack(mat), cols


def _circuit_center(train: pd.DataFrame) -> tuple[dict, float]:
    means = train.groupby("circuit")["LapTime"].mean().to_dict()
    return means, float(train["LapTime"].mean())


def fit_main(train: pd.DataFrame) -> MainModel:
    cmean, gmean = _circuit_center(train)
    seasons = sorted(train["season"].unique().tolist())
    y = train["LapTime"].to_numpy() - train["circuit"].map(cmean).to_numpy()
    X, cols = _design(train, seasons)
    groups = train["driver_season"].to_numpy()

    # try MixedLM (random intercept by driver-season)
    try:
        md = sm.MixedLM(y, X, groups=groups, exog_re=np.ones((len(y), 1)))
        mdf = md.fit(method="lbfgs", maxiter=200, disp=False)
        if not mdf.converged:
            raise RuntimeError("MixedLM did not converge")
        re = {g: float(np.asarray(v).ravel()[0]) for g, v in mdf.random_effects.items()}
        return MainModel(cmean, gmean, seasons, cols, np.asarray(mdf.fe_params, float),
                         re, True, "MixedLM")
    except Exception:
        pass

    # fallback: OLS with driver fixed effects
    dummies = pd.get_dummies(train["driver_season"], prefix="drv", drop_first=True)
    Xf = np.column_stack([X, dummies.to_numpy(float)])
    res = sm.OLS(y, Xf).fit()
    p = np.asarray(res.params, float)
    fe = p[: len(cols)]
    driver_fe = dict(zip(dummies.columns, p[len(cols):]))
    return MainModel(cmean, gmean, seasons, cols, fe, {}, False, "OLS_driver_FE",
                     driver_fe=driver_fe)


def predict_main(m: MainModel, test: pd.DataFrame) -> np.ndarray:
    X, _ = _design(test, m.seasons)
    yc = X @ m.fe_params
    if m.method == "MixedLM":
        ds = test["driver_season"].to_numpy()
        yc = yc + np.array([m.re.get(g, 0.0) for g in ds])
    else:
        ds = test["driver_season"].to_numpy()
        yc = yc + np.array([m.driver_fe.get(f"drv_{g}", 0.0) for g in ds])
    base = test["circuit"].map(m.circuit_mean).fillna(m.global_mean).to_numpy()
    return yc + base
