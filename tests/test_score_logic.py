"""Scoring transforms — hermetic (fake models, no frozen artifacts needed),
plus an integration check on the data-derived positional premiums.

  _tier               score band -> tier, boundary-exact
  _percentile_scores  positional value premiums (where league format enters
                      the cross-position score) and the 1..100 mapping
  _premiums           derived from season_ppg: QB > 1 (superflex), non-QB mean 1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.model import score
from src.util import ROOT, config

PREMS = {"QB": 1.3, "WR": 1.0}  # explicit premiums for the hermetic tests


class _Fake:
    """Stand-in model: predict_proba returns a fixed hit prob for every row."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _frozen(p=0.5):
    tp = np.linspace(0.05, 0.6, 50)
    return {pos: {"model": _Fake(p), "cols": ["log_pick"], "full_lr": _Fake(p),
                  "full_cols": ["log_pick"], "train_phit": tp}
            for pos in ["QB", "WR"]}


def test_tier_bands_are_boundary_exact():
    bands = config()["board"]["tiers"]  # [{min:88,tier:1}, ...74->2, 58->3, 40->4, 0->5]
    assert [score._tier(s) for s in (100, 88)] == [1, 1]
    assert score._tier(87) == 2 and score._tier(74) == 2
    assert score._tier(58) == 3
    assert score._tier(40) == 4
    assert score._tier(39) == 5 and score._tier(0) == 5
    # every band's own min maps to that band
    for b in bands:
        assert score._tier(b["min"]) == b["tier"]


def test_premium_lifts_qb_over_equal_wr():
    out = score._percentile_scores(
        pd.DataFrame({"position": ["QB", "WR"], "log_pick": [1.0, 1.0]}),
        _frozen(0.5), PREMS)
    qb = out.loc[out.position == "QB", "score_1_100"].iloc[0]
    wr = out.loc[out.position == "WR", "score_1_100"].iloc[0]
    # identical raw P(hit); QB's higher positional premium must score it higher.
    assert qb > wr, f"QB {qb} !> WR {wr} despite premium"
    assert out.p_hit.tolist() == [0.5, 0.5]          # raw P(hit) unchanged by premium


def test_scores_stay_in_1_100():
    for p in (0.0, 0.5, 1.0):
        out = score._percentile_scores(
            pd.DataFrame({"position": ["QB", "WR"], "log_pick": [1.0, 1.0]}),
            _frozen(p), PREMS)
        assert out.score_1_100.between(1, 100).all()


def test_lean_is_full_minus_capital():
    # full model hotter than capital -> positive lean, and vice versa.
    fr = _frozen()
    fr["WR"]["full_lr"] = _Fake(0.8)   # full > capital(0.5)
    fr["QB"]["full_lr"] = _Fake(0.2)   # full < capital(0.5)
    out = score._percentile_scores(
        pd.DataFrame({"position": ["QB", "WR"], "log_pick": [1.0, 1.0]}), fr, PREMS)
    assert out.loc[out.position == "WR", "lean"].iloc[0] > 0
    assert out.loc[out.position == "QB", "lean"].iloc[0] < 0


def test_premiums_normalization():
    """Hermetic: normalization keeps non-QB mean at 1.0."""
    prems = score._premiums({"QB": 7.0, "RB": 4.5, "WR": 4.5, "TE": 2.25})
    non_qb = [v for p, v in prems.items() if p != "QB"]
    assert abs(np.mean(non_qb) - 1.0) < 0.01
    assert prems["QB"] == round(7.0 / 3.75, 3)


def test_derived_hit_vor():
    """Integration: hit VOR derived from built season_ppg behaves sanely."""
    if not (ROOT / "data/interim/season_ppg.parquet").exists():
        import pytest  # noqa: PLC0415
        pytest.skip("season_ppg not built; run python -m src.labels.build_labels")
    vor = score._hit_vor()
    assert set(vor) == {"QB", "RB", "WR", "TE"}
    assert all(v > 0 for v in vor.values())
    prems = score._premiums(vor)
    assert prems["QB"] > 1.3, prems            # superflex: QB hits worth the most
    assert prems["TE"] < 1.0 < prems["QB"]     # TE hits worth the least
