"""Continuous-target model: expected 3-season value (VOR) per rookie.

Target `value` (src.labels.build_labels) = mean of a player's best 2 seasons of
ppg over the lineup replacement level in his first 3 NFL seasons, in this
league's scoring. It replaces the binary 'hit' as the score's basis: a binary
label wastes most of ~870 samples and let one 8-game RB fill-in count fully.

Models per position, all walk-forward with the label-realizability embargo
(score class v trains only on classes <= v - window):
  capital     smoothed isotonic regression of value on pick (monotone: better
              pick, higher value; no functional form). The bar to clear.
  capital_age capital + ridge on its residual using age only
  full        capital + ridge on its residual using every feature

Residual stacking keeps capital in charge: challengers can only *adjust* the
draft-slot expectation, and the adjustment is exactly the board's 'lean'.

Ship rule: a challenger replaces capital only if it improves Spearman AND RMSE
out of class in >=90% of paired bootstrap resamples. Either way the best
challenger's adjustment is exported as 'lean', tagged validated or not.

Also derived here (historical data only, no 2026 outputs used):
  horizon     per-position multiplier (1 + years 4-6 VOR / years 1-3 VOR) for
              cross-position ranking: RB value is front-loaded, WR/TE back-loaded
  intervals   80% ranges of realized value given predicted value (OOF)

Run:  python -m src.model.value   (after src.features.build)
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model.features_spec import full_features
from src.util import ROOT, config

warnings.filterwarnings("ignore")
SEED = config()["seed"]
PROC = ROOT / "data/processed"
CONFIDENCE = 0.9
RIDGE_ALPHA = 10.0   # strong shrinkage: ~125-350 rows per position
AGE_COLS = ["age"]
N_BINS = 5           # predicted-value bins for the 80% ranges
HORIZON_SEASONS = 6


def _X(df: pd.DataFrame, cols: list) -> np.ndarray:
    # combine + age are imputed upstream; remaining NaNs are z-scored college
    # shares (missing college) -> 0 = position average
    return df[cols].astype(float).fillna(0.0).values


class StackedValue:
    """Smoothed-isotonic capital curve + optional ridge on its residual.

    Capital curve: isotonic regression of value on pick (monotone, no
    functional form), then a PCHIP spline through each pooled block's mean
    log-pick. Raw isotonic is a step function that ties whole runs of picks
    (e.g. WRs 71-75) -- useless for ordering a board. PCHIP keeps it monotone
    and validated as well or better out of class at RB/WR/TE (QB -0.02 rho).
    Outside the training pick range it is held flat (no extrapolation).
    """

    def __init__(self, cols: list | None):
        self.cols = cols
        self.ridge = (make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
                      if cols else None)

    def _fit_capital(self, pick: np.ndarray, y: np.ndarray) -> None:
        fitted = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(pick, y).predict(pick)
        blk = (pd.DataFrame({"lp": np.log(pick), "f": fitted})
               .groupby("f").lp.mean().reset_index().sort_values("lp"))
        self._lp_range = (float(blk.lp.min()), float(blk.lp.max()))
        self._flat = float(fitted.mean()) if len(blk) < 2 else None
        if self._flat is None:
            self._curve = PchipInterpolator(blk.lp.values, blk.f.values, extrapolate=False)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "StackedValue":
        self._fit_capital(df.pick.values.astype(float), y)
        if self.ridge is not None:
            self.ridge.fit(_X(df, self.cols), y - self.capital(df))
        return self

    def capital(self, df: pd.DataFrame) -> np.ndarray:
        if self._flat is not None:
            return np.full(len(df), self._flat)
        lp = np.clip(np.log(df.pick.values.astype(float)), *self._lp_range)
        return np.clip(self._curve(lp), 0, None)

    def adjustment(self, df: pd.DataFrame) -> np.ndarray:
        if self.ridge is None:
            return np.zeros(len(df))
        return self.ridge.predict(_X(df, self.cols))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # value is floored at 0, so the prediction is too
        return np.clip(self.capital(df) + self.adjustment(df), 0, None)

    def contributions(self, df: pd.DataFrame) -> np.ndarray:
        """Per-feature additive contributions to the adjustment (ppg VOR)."""
        sc, rg = self.ridge.named_steps["standardscaler"], self.ridge.named_steps["ridge"]
        return ((_X(df, self.cols) - sc.mean_) / sc.scale_) * rg.coef_


RUNS = {"capital": lambda pos: None,
        "capital_age": lambda pos: AGE_COLS,
        # log_pick is left out: the isotonic curve already owns draft slot, so
        # the residual model is purely "what production/age add beyond it"
        "full": lambda pos: [c for c in full_features(pos) if c != "log_pick"]}


def _oof(d: pd.DataFrame, cols) -> pd.Series:
    embargo = config()["labels"]["window_seasons"]
    out = pd.Series(np.nan, index=d.index)
    for v in sorted(d.draft_season.unique()):
        tr, va = d[d.draft_season <= v - embargo], d[d.draft_season == v]
        if va.empty or len(tr) < 30:
            continue
        out.loc[va.index] = StackedValue(cols).fit(tr, tr.value.values).predict(va)
    return out


def _metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    return float(spearmanr(p, y).correlation), float(np.sqrt(np.mean((p - y) ** 2)))


def _bootstrap(y: np.ndarray, preds: dict, B: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(y)
    per = {k: {"spearman": [], "rmse": []} for k in preds}
    beats = {k: [] for k in preds if k != "capital"}
    for _ in range(B):
        s = rng.integers(0, n, n)
        if np.ptp(y[s]) == 0:
            continue
        m = {k: _metrics(y[s], p[s]) for k, p in preds.items()}
        for k, (sp, rm) in m.items():
            per[k]["spearman"].append(sp); per[k]["rmse"].append(rm)
        for k in beats:
            beats[k].append(m[k][0] > m["capital"][0] and m[k][1] < m["capital"][1])

    def ci(v):
        return [round(float(np.nanpercentile(v, 10)), 3), round(float(np.nanpercentile(v, 90)), 3)]

    out = {k: {"spearman_ci": ci(v["spearman"]), "rmse_ci": ci(v["rmse"])} for k, v in per.items()}
    for k, v in beats.items():
        out[k]["p_beats_capital"] = round(float(np.mean(v)), 3)
    return out


def _intervals(pred: np.ndarray, y: np.ndarray) -> dict:
    """80% range of realized value within predicted-value quantile bins."""
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, N_BINS + 1)))
    idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)
    lo, hi = [], []
    for b in range(len(edges) - 1):
        yb = y[idx == b]
        lo.append(float(np.percentile(yb, 10)) if len(yb) else 0.0)
        hi.append(float(np.percentile(yb, 90)) if len(yb) else 0.0)
    return {"edges": edges.tolist(), "lo": lo, "hi": hi}


def interval_for(iv: dict, pred: float) -> tuple[float, float]:
    edges = np.asarray(iv["edges"])
    b = int(np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(iv["lo"]) - 1))
    return iv["lo"][b], iv["hi"][b]


def horizon_multipliers() -> dict:
    """{pos: 1 + (years 4-6 VOR) / (years 1-3 VOR)} over classes old enough to
    have 6 observed seasons. Converts 3-season value into a 6-season dynasty
    horizon so positions with different aging curves compare fairly."""
    cfg = config()
    ppg = pd.read_parquet(ROOT / "data/interim/season_ppg.parquet")
    vor = ppg.groupby(["player_id", "season"]).vor.sum().to_dict()
    last = int(ppg.season.max()) - HORIZON_SEASONS + 1
    d = pd.read_parquet(ROOT / "data/raw/nfl_draft.parquet")
    d = d[d.position.isin(cfg["positions"]) & d.season.between(cfg["seasons"]["nfl_start"], last)
          & d.gsis_id.notna()]
    w = cfg["labels"]["window_seasons"]
    early, late = {}, {}
    for r in d.itertuples():
        s = [vor.get((r.gsis_id, r.season + k), 0.0) for k in range(HORIZON_SEASONS)]
        early[r.position] = early.get(r.position, 0.0) + sum(s[:w])
        late[r.position] = late.get(r.position, 0.0) + sum(s[w:])
    return {p: round(float(1 + late[p] / early[p]), 3) for p in early}


def run() -> dict:
    f = pd.read_parquet(ROOT / "data/interim/features.parquet")
    lab = pd.read_parquet(ROOT / "data/interim/labels.parquet")[["gsis_id", "value"]]
    f = f[f.label_complete.astype(bool)].merge(lab, on="gsis_id", how="left")
    f = f[f.value.notna()].copy()

    horizon = horizon_multipliers()
    metrics, frozen = {"horizon": horizon}, {}
    for pos in config()["positions"]:
        d = f[f.position == pos].reset_index(drop=True)
        oof = {k: _oof(d, cols(pos)) for k, cols in RUNS.items()}
        ok = oof["capital"].notna().values
        y = d.value.values[ok]
        preds = {k: v.values[ok] for k, v in oof.items()}
        m = {k: dict(zip(("spearman", "rmse"), np.round(_metrics(y, p), 3))) for k, p in preds.items()}
        for k, b in _bootstrap(y, preds).items():
            m[k].update(b)
            m[k]["n"] = int(ok.sum())
        winners = [k for k in ("capital_age", "full")
                   if m[k]["spearman"] > m["capital"]["spearman"]
                   and m[k]["rmse"] < m["capital"]["rmse"]
                   and m[k]["p_beats_capital"] >= CONFIDENCE]
        chosen = max(winners, key=lambda k: m[k]["spearman"]) if winners else "capital"
        # the lean comes from whichever challenger did best, validated or not
        # lean always comes from the full residual model: its job is to show
        # where production/age disagree with draft slot, validated or not
        lean_from = "full"
        m["chosen"], m["lean_from"] = chosen, lean_from
        m["lean_validated"] = lean_from in winners
        metrics[pos] = m

        models = {k: StackedValue(RUNS[k](pos)).fit(d, d.value.values) for k in RUNS}
        hist = np.clip(models[chosen].predict(d), 0, None) * horizon[pos]
        frozen[pos] = {"models": models, "chosen": chosen, "lean_from": lean_from,
                       "lean_validated": m["lean_validated"],
                       "intervals": _intervals(preds[chosen], y),
                       "hist_value": hist, "horizon": horizon[pos],
                       "cap_mean": float(models["capital"].capital(d).mean()),
                       "reliability": m[chosen]["spearman"],
                       "reliability_ci": m[chosen]["spearman_ci"],
                       "lean_p_beats": m["full"]["p_beats_capital"]}

    joblib.dump(frozen, PROC / "value_models.joblib")
    (PROC / "value_metrics.json").write_text(json.dumps(metrics, indent=2))
    _report(metrics)
    return metrics


def _report(metrics: dict) -> None:
    print(f"horizon multipliers: {metrics['horizon']}")
    print(f"{'pos':4} {'model':12} {'n':>4} {'spear':>6} {'80% CI':>15} {'rmse':>6} {'p>cap':>6}")
    for pos in config()["positions"]:
        m = metrics[pos]
        for k in RUNS:
            e = m[k]
            tag = " <-- chosen" if k == m["chosen"] else (" (lean)" if k == m["lean_from"] else "")
            pb = f"{e.get('p_beats_capital', ''):>6}"
            print(f"{pos:4} {k:12} {e['n']:>4} {e['spearman']:>6} {str(e['spearman_ci']):>15} "
                  f"{e['rmse']:>6} {pb}{tag}")
        print()


if __name__ == "__main__":
    # run via the package module so pickled StackedValue resolves as
    # src.model.value.StackedValue, not __main__.StackedValue
    from src.model import value as _value
    _value.run()
