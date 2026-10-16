"""Board checks: data contract, score range/order, tiers, driver shape."""
from __future__ import annotations

import json

from src.util import ROOT

BOARD = ROOT / "data/processed/board.json"
CONTRACT = {"player_id", "name", "pos", "school", "pick", "round", "age",
            "score_1_100", "p_hit", "tier", "drivers", "ktc_rank",
            "delta_vs_ktc", "college_stats", "exp_vor", "dyn_value", "value_range",
            "lean", "lean_validated", "reliability"}


def _board():
    if not BOARD.exists():
        import pytest  # noqa: PLC0415
        pytest.skip("board not built; run python -m src.model.score")
    return json.loads(BOARD.read_text())


def test_data_contract_keys():
    b = _board()
    assert b, "board is empty"
    for row in b:
        assert CONTRACT <= set(row), f"missing keys: {CONTRACT - set(row)}"


def test_scores_in_range_and_sorted():
    b = _board()
    scores = [r["score_1_100"] for r in b]
    assert all(1 <= s <= 100 for s in scores)
    assert scores == sorted(scores, reverse=True), "board must be sorted by score desc"


def test_no_imputation_artifacts_in_drivers():
    """Regression for the NaN -> 0 bugs (combine, age): a blank read as 0
    showed up as |z| of 9-25. Real extremes top out near 5."""
    for r in _board():
        for d in r["drivers"]:
            assert d["z"] is None or abs(d["z"]) < 6, (r["name"], d)
        assert r["age"] is not None, r["name"]


def test_value_fields_consistent():
    for r in _board():
        lo, hi = r["value_range"]
        assert 0 <= lo <= hi
        assert r["dyn_value"] >= r["exp_vor"] >= 0


def test_tiers_and_positions_valid():
    b = _board()
    for r in b:
        assert r["tier"] in {1, 2, 3, 4, 5}
        assert r["pos"] in {"QB", "RB", "WR", "TE"}


def test_drivers_shape():
    b = _board()
    for r in b:
        assert 1 <= len(r["drivers"]) <= 3
        for d in r["drivers"]:
            assert set(d) == {"name", "direction", "magnitude", "value", "z"}
            assert d["direction"] in {"positive", "negative"}
            assert d["value"] is None or isinstance(d["value"], (int, float))
        for c in r.get("comps", []):
            assert set(c) == {"name", "year", "pick", "hit", "peak", "why"}
            assert isinstance(c["hit"], bool) and 1 <= len(c["why"]) <= 2


