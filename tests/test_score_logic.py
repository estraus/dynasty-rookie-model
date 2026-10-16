"""Scoring transforms -- hermetic (synthetic data, no frozen artifacts needed).

  _tier             score band -> tier, boundary-exact
  player_value      continuous label: best-2-of-3 season VOR, missing seasons = 0
  StackedValue      monotone capital curve; residual adjustment = lean
  interval_for      80% range lookup
  _value_scores     horizon multiplier shifts cross-position order; 1..100 range
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels.build_labels import player_value
from src.model import score
from src.model.value import StackedValue, _intervals, interval_for
from src.util import config


def test_tier_bands_are_boundary_exact():
    bands = config()["board"]["tiers"]
    assert [score._tier(s) for s in (100, 88)] == [1, 1]
    assert score._tier(87) == 2 and score._tier(74) == 2
    assert score._tier(58) == 3
    assert score._tier(40) == 4
    assert score._tier(39) == 5 and score._tier(0) == 5
    for b in bands:
        assert score._tier(b["min"]) == b["tier"]


def test_player_value_best_two_of_window():
    assert player_value([6.0], 3) == 3.0              # one hot season is worth half
    assert player_value([6.0, 6.0], 3) == 6.0         # repeated, it's worth full
    assert player_value([1.0, 5.0, 3.0], 3) == 4.0    # best two of three
    assert player_value([], 3) == 0.0


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.integers(1, 260, n)
    age = rng.normal(22, 1, n)
    value = np.clip(8 * np.exp(-pick / 40) + rng.normal(0, 0.5, n), 0, None)
    return pd.DataFrame({"pick": pick, "age": age, "value": value})


def test_capital_curve_is_monotone_and_smooth():
    d = _synthetic()
    m = StackedValue(None).fit(d, d.value.values)
    grid = pd.DataFrame({"pick": np.arange(1, 263)})
    cap = m.capital(grid)
    assert np.all(np.diff(cap) <= 1e-9), "better pick must never be worth less"
    assert len(np.unique(np.round(cap, 6))) > 50, "curve should not be a step function"
    assert np.all(m.adjustment(grid) == 0)            # capital-only: no lean


def test_residual_adjustment_is_the_lean():
    d = _synthetic()
    d["value"] = d.value + 0.8 * (22 - d.age)          # younger -> more value
    m = StackedValue(["age"]).fit(d, d.value.values)
    young, old = (pd.DataFrame({"pick": [50], "age": [a]}) for a in (20.5, 23.5))
    assert m.adjustment(young)[0] > 0 > m.adjustment(old)[0]
    assert m.predict(young)[0] > m.predict(old)[0]


def test_interval_lookup():
    pred = np.linspace(0, 5, 100)
    iv = _intervals(pred, pred * 1.0)
    lo, hi = interval_for(iv, 4.9)
    assert lo <= 4.9 <= hi + 1e-9
    assert interval_for(iv, -1)[0] == iv["lo"][0]     # clamps below range


class _Hit:
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.6), np.full(len(X), 0.4)])


def test_horizon_multiplier_orders_positions():
    d = _synthetic()
    models = {k: StackedValue(None).fit(d, d.value.values) for k in ("capital", "full")}
    vf = {pos: {"models": models, "chosen": "capital", "lean_from": "full",
                "intervals": _intervals(np.linspace(0, 8, 50), np.linspace(0, 8, 50)),
                "hist_value": np.linspace(0, 16, 200), "horizon": h,
                "lean_validated": False, "reliability": 0.5}
          for pos, h in (("RB", 1.8), ("WR", 2.0))}
    bf = {pos: {"model": _Hit(), "cols": ["log_pick"]} for pos in vf}
    feats = pd.DataFrame({"position": ["RB", "WR"], "pick": [30, 30], "log_pick": np.log([30, 30])})
    out = score._value_scores(feats, vf, bf)
    assert out.exp_vor.iloc[0] == out.exp_vor.iloc[1]          # same slot, same 3yr value
    wr, rb = out.set_index("position").score_1_100[["WR", "RB"]]
    assert wr > rb, "longer WR horizon must lift an equal-value WR"
    assert out.score_1_100.between(1, 100).all()
    assert (out.range_lo <= out.range_hi).all()
