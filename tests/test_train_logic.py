"""Model-selection + bootstrap logic — hermetic (no data/model caches).

Covers the ship rule (_choose) and the paired bootstrap (_bootstrap): a
challenger ships only if it beats the capital baseline on point estimate AND
the win survives resampling. These gate whether complexity leaves the lab.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import train


def _m(auc, brier, p_beats=None):
    d = {"auc": auc, "brier": brier}
    if p_beats is not None:
        d["p_beats_baseline"] = p_beats
    return d


def test_choose_baseline_when_no_point_win():
    m = {"baseline": _m(0.80, 0.10),
         "full_lr": _m(0.78, 0.11, 0.9),      # worse auc
         "full_lgbm": _m(0.81, 0.12, 0.9)}    # better auc but worse brier
    assert train._choose(m) == "baseline"


def test_choose_baseline_when_win_is_inside_noise():
    # point win on both, but the bootstrap says it only holds 60% of the time.
    m = {"baseline": _m(0.80, 0.10),
         "full_lr": _m(0.84, 0.09, 0.60),
         "full_lgbm": _m(0.82, 0.095, 0.50)}
    assert train._choose(m) == "baseline"


def test_choose_challenger_when_win_is_real():
    m = {"baseline": _m(0.80, 0.10),
         "full_lr": _m(0.84, 0.092, 0.95),    # real win, brier 0.092
         "full_lgbm": _m(0.85, 0.088, 0.93)}  # real win, better brier 0.088
    assert train._choose(m) == "full_lgbm"    # tie-break: best (lowest) brier


def test_choose_handles_degenerate_metrics():
    m = {"baseline": _m(None, None), "full_lr": _m(None, None),
         "full_lgbm": _m(None, None)}
    assert train._choose(m) == "baseline"


def _series(vals):
    return pd.Series(vals, dtype=float)


def test_bootstrap_separating_model_beats_random():
    # baseline perfectly ranks y; full_lr is uninformative (constant 0.5).
    y = _series([0, 0, 0, 0, 1, 1, 1, 1] * 6)
    p_sep = _series([0.1, 0.15, 0.2, 0.25, 0.75, 0.8, 0.85, 0.9] * 6)
    p_rand = _series([0.5] * 48)
    out = train._bootstrap(y, {"baseline": p_sep, "full_lr": p_rand}, B=300)
    lo, hi = out["baseline"]["auc_ci"]
    assert lo > 0.9 and hi <= 1.0                 # separating model: tight, high
    # the random challenger essentially never beats the separating baseline
    assert out["full_lr"]["p_beats_baseline"] < 0.05


def test_bootstrap_ci_is_ordered_and_bounded():
    y = _series([0, 1] * 24)
    p = _series(np.linspace(0.2, 0.8, 48))
    out = train._bootstrap(y, {"baseline": p}, B=200)
    lo, hi = out["baseline"]["auc_ci"]
    assert 0.0 <= lo <= hi <= 1.0
