"""Label checks: scoring math (the money path) + known hit/bust anchors."""
from __future__ import annotations

import pandas as pd

from src.labels.build_labels import score_weekly
from src.util import ROOT, config

LABELS = ROOT / "data/interim/labels.parquet"

ALL_STAT_COLS = [
    "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
]


def _row(position="WR", **kw):
    base = {c: 0 for c in ALL_STAT_COLS} | {"position": position}
    base.update(kw)
    return base


def test_scoring_math():
    # QB line: 300 pass yds, 3 pass TD, 1 INT, 20 rush yds, 1 rush TD
    #   = 300*.04 + 3*6 - 1*2 + 20*.1 + 1*6 = 12+18-2+2+6 = 36.0
    # WR line (half-PPR): 5 rec, 50 yds, 1 TD, 1 lost fumble
    #   = 5*.5 + 50*.1 + 6 - 1*2 = 2.5+5+6-2 = 11.5
    df = pd.DataFrame([
        _row(passing_yards=300, passing_tds=3, interceptions=1, rushing_yards=20, rushing_tds=1),
        _row(receptions=5, receiving_yards=50, receiving_tds=1, receiving_fumbles_lost=1),
    ])
    pts = score_weekly(df).tolist()
    assert pts[0] == 36.0, pts[0]
    assert pts[1] == 11.5, pts[1]


def test_half_ppr_reception_value():
    # a reception must be worth exactly 0.5 (half-PPR)
    delta = score_weekly(pd.DataFrame([_row(receptions=1)])).iloc[0]
    assert delta == 0.5


def test_te_reception_premium():
    # league TEP (bonus_rec_te 0.25): TE reception = 0.75, others 0.5;
    # market convention (te_reception_bonus 0) must not apply it
    df = pd.DataFrame([_row(position="TE", receptions=1), _row(receptions=1)])
    assert score_weekly(df).tolist() == [0.75, 0.5]
    mkt = {**config()["league"], "te_reception_bonus": 0.0}
    assert score_weekly(df, mkt).tolist() == [0.5, 0.5]


def _labels():
    if not LABELS.exists():
        import pytest  # noqa: PLC0415
        pytest.skip("labels not built; run python -m src.labels.build_labels")
    return pd.read_parquet(LABELS)


def test_known_hits_and_busts():
    L = _labels()
    hits = ["Saquon Barkley", "Justin Jefferson", "Patrick Mahomes", "Lamar Jackson"]
    busts = ["EJ Manuel", "Zach Wilson", "N'Keal Harry", "Johnny Manziel"]
    for n in hits:
        assert bool(L.loc[L.player == n, "hit"].iloc[0]), f"{n} should be a hit"
    for n in busts:
        assert not bool(L.loc[L.player == n, "hit"].iloc[0]), f"{n} should be a bust"


def test_incomplete_windows_flagged():
    L = _labels()
    # 2023 class cannot have a complete 3yr window (weekly data ends 2024)
    assert not L[L.draft_season == 2023].label_complete.any()
    assert L[L.draft_season <= 2022].label_complete.all()


