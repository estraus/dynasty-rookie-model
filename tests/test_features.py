"""Feature checks pinned to known players (plan acceptance) + share math.

Rebuilds the college table (fast, from cached parquet) so these are hermetic
given the raw caches; the feature-matrix anchors read the built parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import college
from src.util import ROOT

FEATURES = ROOT / "data/interim/features.parquet"
HOPKINS_CFBD = "501689"  # DeAndre Hopkins, Clemson


def _col():
    if not hasattr(_col, "cache"):
        _col.cache = college.build()
    return _col.cache


def test_receiving_not_colliding_with_rushing():
    """Regression: category prefix bug merged receiving YDS into rushing YDS."""
    s = college.player_seasons(_col(), HOPKINS_CFBD)
    y2012 = s[s.season == 2012].iloc[0]
    assert y2012.rec_yds == 1405, y2012.rec_yds       # receiving preserved
    assert y2012.rec == 82
    assert pd.isna(y2012.rush_yds) or y2012.rush_yds == 0  # Hopkins didn't rush


def test_hopkins_dominator_by_hand():
    s = college.player_seasons(_col(), HOPKINS_CFBD)
    r = s[s.season == 2012].iloc[0]
    expect = 0.5 * (r.rec_yds / r.team_pass_yds) + 0.5 * (r.rec_td / r.team_pass_td)
    assert abs(r.dominator - expect) < 1e-9
    assert abs(r.dominator - 0.3930) < 1e-3       # pinned value
    assert r.z_dominator > 2                       # elite -> high within-season z


def test_speed_score_formula():
    # speed score = (weight * 200) / (40 time ^ 4)
    from src.features.build import _combine
    cb = _combine()
    row = cb.dropna(subset=["speed_score"]).iloc[0]
    assert abs(row.speed_score - (row.wt * 200 / row.forty**4)) < 1e-6


def _features():
    if not FEATURES.exists():
        import pytest  # noqa: PLC0415
        pytest.skip("features not built; run python -m src.features.build")
    return pd.read_parquet(FEATURES)


def test_known_player_features():
    f = _features().set_index("player")
    hop = f.loc["DeAndre Hopkins"]
    assert hop.power_conf == 1 and hop.early_declare == 1
    assert hop.wr_breakout_age <= 20            # freshman/soph breakout
    lam = f.loc["Lamar Jackson"]
    assert lam.qb_rush_ypg > 80                  # elite college rusher
    assert lam.qb_z_ypa > 0


def test_combine_missingness_flagged_and_imputed():
    f = _features()
    assert f.speed_score.notna().all()           # all imputed
    assert set(f.combine_missing.unique()) <= {0, 1}
    assert f.combine_missing.mean() > 0           # some were genuinely missing


