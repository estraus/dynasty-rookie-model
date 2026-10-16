"""Hermetic checks for the value model and the in-season update (synthetic data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model.inseason import _prior_only, features_at, percentile_score, predict
from src.model.value import StackedValue, interval_for


def _picks(n=80, seed=0):
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.integers(1, 260, n)).astype(float)
    value = np.clip(8 - 1.4 * np.log(pick) + rng.normal(0, 1, n), 0, None)
    return pd.DataFrame({"pick": pick, "age": rng.normal(22, 1, n)}), value


def test_capital_curve_is_monotone_and_nonnegative():
    df, y = _picks()
    m = StackedValue(None).fit(df, y)
    grid = pd.DataFrame({"pick": np.arange(1, 263, dtype=float)})
    cap = m.capital(grid)
    assert (np.diff(cap) <= 1e-9).all()      # better pick never worth less
    assert (cap >= 0).all()


def test_capital_only_has_zero_adjustment():
    df, y = _picks()
    m = StackedValue(None).fit(df, y)
    assert np.allclose(m.adjustment(df), 0.0)
    assert np.allclose(m.predict(df), m.capital(df))


def test_interval_lookup_clamps_to_bins():
    iv = {"edges": [0.0, 1.0, 2.0], "lo": [0.0, 0.5], "hi": [1.0, 3.0]}
    assert interval_for(iv, -5) == (0.0, 1.0)
    assert interval_for(iv, 1.5) == (0.5, 3.0)
    assert interval_for(iv, 99) == (0.5, 3.0)


def _usage():
    rows = [("p1", 2026, w, "WR", 20.0, 0.3, 0.0) for w in range(1, 11)]
    return pd.DataFrame(rows, columns=["player_id", "season", "week", "position",
                                       "pts", "tgt_share", "carry_share"])


PLAYERS = pd.DataFrame({"gsis_id": ["p1", "p2"], "position": ["WR", "WR"],
                        "draft_season": [2026, 2026], "prior": [2.0, 1.0]})
REPL = {(2025, "WR"): 8.0, (2026, "WR"): 8.0}


def test_inseason_features_ignore_weeks_after_checkpoint():
    """Week 10 exists in the data; a week-6 checkpoint must not see it."""
    f = features_at(PLAYERS, _usage(), REPL, seasons_done=0, week=6)
    p1 = f[f.gsis_id == "p1"].iloc[0]
    assert p1.games == 6
    assert np.isclose(p1.vor_pg, 20.0 - 8.0)


def test_no_show_reads_as_minus_replacement():
    f = features_at(PLAYERS, _usage(), REPL, seasons_done=0, week=6)
    p2 = f[f.gsis_id == "p2"].iloc[0]
    assert p2.games == 0 and np.isclose(p2.vor_pg, -8.0)


def test_prior_only_model_returns_draft_value():
    X = features_at(PLAYERS, _usage(), REPL, seasons_done=0, week=6)
    assert np.allclose(predict(_prior_only(), X), X.prior.values)


def test_percentile_score_bounds():
    hist = np.sort(np.arange(100, dtype=float))
    assert percentile_score(np.array([-1.0, 1e9]), hist).tolist() == [1, 100]
