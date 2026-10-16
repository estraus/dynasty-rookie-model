"""Crosswalk integration check. Runs against the built crosswalk parquet.

Skips if the crosswalk hasn't been built yet (fresh clone without data caches).
Run:  ./.venv/bin/python -m pytest tests/test_crosswalk.py
"""
from __future__ import annotations

import pandas as pd

from src.util import ROOT

XWALK = ROOT / "data/interim/crosswalk.parquet"

# Anchors: name -> (must_match?, expected matched_name substring or None)
ANCHORS = {
    "DeAndre Hopkins": (True, "Hopkins"),
    "Saquon Barkley": (True, "Barkley"),
    "Patrick Mahomes": (True, "Mahomes"),
    "Tank Dell": (True, "Dell"),          # nickname recovery -> Nathaniel Dell
    "Chig Okonkwo": (True, "Okonkwo"),    # nickname recovery -> Chigoziem Okonkwo
    "Todd Gurley": (False, None),         # genuine CFBD gap -> must be a miss, not a wrong match
}


def _load() -> pd.DataFrame:
    if not XWALK.exists():
        import pytest  # noqa: PLC0415
        pytest.skip("crosswalk not built; run python -m src.ids.crosswalk")
    return pd.read_parquet(XWALK)


def test_match_rate_at_least_95pct():
    x = _load()
    rate = x.cfbd_player_id.notna().mean()
    assert rate >= 0.95, f"match rate {rate:.3f} below 0.95"


def test_no_college_player_matched_twice():
    """Two different draftees mapping to one cfbd_player_id => a false positive."""
    x = _load()
    matched = x[x.cfbd_player_id.notna()]
    dupes = matched.cfbd_player_id.value_counts()
    dupes = dupes[dupes > 1]
    assert dupes.empty, f"cfbd_player_id reused across draftees: {dupes.to_dict()}"


def test_anchors():
    x = _load()
    for name, (must_match, sub) in ANCHORS.items():
        r = x[x.pfr_player_name == name]
        assert len(r) == 1, f"{name}: expected exactly one draft row, got {len(r)}"
        row = r.iloc[0]
        if must_match:
            assert pd.notna(row.cfbd_player_id), f"{name} should match, got miss"
            assert sub in row.matched_name, f"{name} matched wrong player {row.matched_name!r}"
        else:
            assert pd.isna(row.cfbd_player_id), (
                f"{name} is a known CFBD gap but matched {row.matched_name!r} "
                f"via {row.match_method} -- likely false positive")


