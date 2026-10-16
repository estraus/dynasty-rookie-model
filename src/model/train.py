"""Train + evaluate: capital-only baseline vs full model, rolling-class CV.

For each position we compare three models on out-of-class (rolling) validation:
  baseline   L2 logistic on log(pick) only  -- the bar to clear
  full_lr    L2 logistic on all features
  full_lgbm  LightGBM, monotone-decreasing in log(pick) (better pick -> higher P)

Ship rule (per plan): the full model must beat the capital-only baseline out of
class, or the honest finding is "capital is enough" -- also valid. We pick, per
position, the challenger with the best out-of-class Brier that also beats
baseline AUC; otherwise we keep baseline. Then freeze on all complete data.

Outputs (data/processed/):
  models.joblib        frozen per-position model + feature list + train P(hit) dist
  metrics.json         per-position, per-model Brier/AUC/Spearman + choice
  calibration.png      reliability curves for the chosen models

Run:  python -m src.model.train
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model.features_spec import BASELINE, full_features
from src.util import ROOT, config

warnings.filterwarnings("ignore")
SEED = config()["seed"]
PROC = ROOT / "data/processed"


def _matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    # z-score NaNs (missing college) -> 0 = league average; combine already imputed
    return df[cols].astype(float).fillna(0.0).values


def _make_lr(cols):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                            random_state=SEED))


def _make_lgbm(cols):
    from lightgbm import LGBMClassifier
    mono = [-1 if c == "log_pick" else 0 for c in cols]  # worse pick -> lower P(hit)
    return LGBMClassifier(
        n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        monotone_constraints=mono, random_state=SEED, verbose=-1)


def _fit(kind, cols, X, y):
    m = _make_lr(cols) if kind == "lr" else _make_lgbm(cols)
    m.fit(X, y)
    return m


def _rolling_oof(df: pd.DataFrame, cols: list, kind: str) -> pd.Series:
    """Out-of-class preds with a label-realizability embargo.

    To score class v at its draft (spring year v), the only classes whose hit
    label is actually known are those whose 3-season window has closed:
    class c is realized once season c+window ends, i.e. c <= v - window. So we
    train on classes <= v - window, NOT <= v-1. Without the embargo the CV trains
    on future-realized outcomes and reports an optimistic metric.
    """
    embargo = config()["labels"]["window_seasons"]
    preds = pd.Series(index=df.index, dtype=float)
    for v in sorted(df.draft_season.unique()):
        tr = df[df.draft_season <= v - embargo]
        va = df[df.draft_season == v]
        if va.empty or tr.y.nunique() < 2 or len(tr) < 30:
            continue
        m = _fit(kind, cols, _matrix(tr, cols), tr.y.values)
        preds.loc[va.index] = m.predict_proba(_matrix(va, cols))[:, 1]
    return preds


def _eval(y, p, outcome) -> dict:
    ok = p.notna()
    y, p, outcome = y[ok], p[ok], outcome[ok]
    if y.nunique() < 2:
        return {"n": int(ok.sum()), "brier": None, "auc": None, "spearman": None}
    return {
        "n": int(ok.sum()),
        "brier": round(brier_score_loss(y, p), 4),
        "auc": round(roc_auc_score(y, p), 3),
        "spearman": round(spearmanr(p, outcome).correlation, 3),
    }


def _bootstrap(y, oof_store: dict, B: int = 2000) -> dict:
    """Percentile CIs per model + paired delta-vs-baseline, resampling players.

    All models share the same out-of-class coverage (fold skips depend only on
    row counts, not features), so a single resample of players scores every
    model on the same draws -- letting us ask 'is the full-vs-baseline gap real?'
    not just 'is the point estimate bigger?'. p_beats_baseline is the fraction of
    resamples where the challenger beats baseline on BOTH auc and brier (the
    _choose rule). 80% CIs (10th-90th pct).

    ponytail: row bootstrap, not fold/class -- with <=8 usable classes a cluster
    bootstrap is too unstable; this slightly understates uncertainty from
    within-class correlation. Raise B / cluster if that ever bites.
    """
    ok = oof_store["baseline"].notna().values
    idx = np.where(ok)[0]
    yv = y.values[idx]
    P = {k: s.values[idx] for k, s in oof_store.items()}
    rng = np.random.default_rng(SEED)
    per = {k: {"auc": [], "brier": []} for k in P}
    delta = {k: {"auc": [], "brier": [], "both": []} for k in P if k != "baseline"}
    n = len(idx)
    for _ in range(B):
        s = rng.integers(0, n, n)
        ys = yv[s]
        if len(np.unique(ys)) < 2:
            continue
        scored = {}
        for k, p in P.items():
            ps = p[s]
            a, b = roc_auc_score(ys, ps), brier_score_loss(ys, ps)
            per[k]["auc"].append(a); per[k]["brier"].append(b)
            scored[k] = (a, b)
        for k in delta:
            da = scored[k][0] - scored["baseline"][0]        # +ve = challenger auc higher
            db = scored["baseline"][1] - scored[k][1]        # +ve = challenger brier lower
            delta[k]["auc"].append(da); delta[k]["brier"].append(db)
            delta[k]["both"].append(da > 0 and db > 0)

    def ci(v):
        return [round(float(np.percentile(v, 10)), 3),
                round(float(np.percentile(v, 90)), 3)] if v else None

    out = {k: {"auc_ci": ci(v["auc"]), "brier_ci": ci(v["brier"])} for k, v in per.items()}
    for k, v in delta.items():
        out[k].update({"d_auc_ci": ci(v["auc"]), "d_brier_ci": ci(v["brier"]),
                       "p_beats_baseline": round(float(np.mean(v["both"])), 3) if v["both"] else 0.0})
    return out


def run() -> dict:
    f = pd.read_parquet(ROOT / "data/interim/features.parquet")
    f = f[f.label_complete].copy()
    f["y"] = f.hit.astype(int)
    # Spearman association of predicted P(hit) with the realized hit outcome
    f["outcome"] = f.y.astype(float)

    PROC.mkdir(parents=True, exist_ok=True)
    metrics, frozen = {}, {}
    calib = {}
    for pos in config()["positions"]:
        d = f[f.position == pos].copy().reset_index(drop=True)
        full = full_features(pos)
        outcome = d.outcome
        runs = {
            "baseline": (BASELINE, "lr"),
            "full_lr": (full, "lr"),
            "full_lgbm": (full, "lgbm"),
        }
        m_pos = {}
        oof_store = {}
        for name, (cols, kind) in runs.items():
            oof = _rolling_oof(d, cols, kind)
            oof_store[name] = oof
            m_pos[name] = _eval(d.y, oof, outcome)

        boot = _bootstrap(d.y, oof_store)
        for name in runs:
            m_pos[name].update(boot[name])
        chosen = _choose(m_pos)
        m_pos["chosen"] = chosen
        metrics[pos] = m_pos
        calib[pos] = _calib_bins(d.y, oof_store[chosen])

        # freeze chosen model on ALL complete data
        cols, kind = runs[chosen]
        model = _fit(kind, cols, _matrix(d, cols), d.y.values)
        train_p = model.predict_proba(_matrix(d, cols))[:, 1]
        # also freeze the full LOGISTIC model (interpretable coefs) for the board's
        # 'lean vs capital' column and top-3 drivers, even though capital-only is
        # the shipped P(hit). LR chosen over LGBM here for per-player attribution.
        full_cols = full_features(pos)
        full_lr = _fit("lr", full_cols, _matrix(d, full_cols), d.y.values)
        frozen[pos] = {"model": model, "cols": cols, "kind": kind,
                       "train_phit": np.sort(train_p),
                       "full_lr": full_lr, "full_cols": full_cols}

    joblib.dump(frozen, PROC / "models.joblib")
    (PROC / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot_calibration(calib, metrics)
    _report(metrics)
    return metrics


CONFIDENCE = 0.9  # bootstrap fraction the win must hold across to ship a challenger


def _choose(m: dict) -> str:
    """Full model must beat baseline out of class AND the gap must be real.

    Point-estimate win on both auc and brier, plus the win holds in >=90% of
    bootstrap resamples (p_beats_baseline). Otherwise 'capital is enough' -- a
    prettier point estimate inside the noise band doesn't earn the complexity.
    """
    base = m["baseline"]
    challengers = {k: m[k] for k in ("full_lr", "full_lgbm")
                   if m[k]["auc"] is not None}
    if base["auc"] is None or not challengers:
        return "baseline"
    better = {k: v for k, v in challengers.items()
              if v["auc"] > base["auc"] and v["brier"] < base["brier"]
              and v.get("p_beats_baseline", 0.0) >= CONFIDENCE}
    if not better:
        return "baseline"
    return min(better, key=lambda k: better[k]["brier"])  # best-calibrated winner


def _calib_bins(y, p, bins=5):
    ok = p.notna()
    y, p = y[ok], p[ok]
    q = pd.qcut(p, min(bins, p.nunique()), duplicates="drop")
    g = pd.DataFrame({"y": y.values, "p": p.values, "q": q.values}).groupby("q", observed=True)
    return {"pred": g.p.mean().tolist(), "obs": g.y.mean().tolist()}


def _plot_calibration(calib, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, pos in zip(axes.flat, config()["positions"]):
        c = calib[pos]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
        ax.plot(c["pred"], c["obs"], "o-", label=metrics[pos]["chosen"])
        ax.set_title(f"{pos}  (chosen: {metrics[pos]['chosen']})")
        ax.set_xlabel("predicted P(hit)"); ax.set_ylabel("observed hit rate")
        ax.legend(fontsize=8)
    fig.suptitle("Out-of-class calibration by position")
    fig.tight_layout()
    fig.savefig(PROC / "calibration.png", dpi=110)
    plt.close(fig)


def _report(metrics):
    print(f"{'pos':4} {'model':10} {'n':>4} {'brier':>7} {'auc':>6} "
          f"{'auc 80% CI':>16} {'p>base':>7} {'spear':>6}")
    for pos, m in metrics.items():
        for name in ("baseline", "full_lr", "full_lgbm"):
            e = m[name]
            star = " <-- chosen" if name == m["chosen"] else ""
            pb = "" if name == "baseline" else f"{e.get('p_beats_baseline', 0):>7}"
            ci = e.get("auc_ci")
            cis = f"[{ci[0]}, {ci[1]}]" if ci else "--"
            print(f"{pos:4} {name:10} {e['n']:>4} {str(e['brier']):>7} "
                  f"{str(e['auc']):>6} {cis:>16} {pb:>7} {str(e['spearman']):>6}{star}")
        print()


if __name__ == "__main__":
    run()
