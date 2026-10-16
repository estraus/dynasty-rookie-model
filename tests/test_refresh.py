"""CI refresh logic — hermetic (no network, no caches)."""
from __future__ import annotations

from src.ingest.refresh import patch_board_ktc


def test_patch_board_ktc_updates_rank_and_delta():
    board = [{"name": "Alpha Guy"}, {"name": "Beta Guy"}, {"name": "Gamma Guy"}]
    out = patch_board_ktc(board, {"alpha guy": 3, "gamma guy": 2})
    # board is score-sorted -> model ranks 1,2,3
    assert out[0]["ktc_rank"] == 3 and out[0]["delta_vs_ktc"] == 2   # market lower = target
    assert out[1]["ktc_rank"] is None and out[1]["delta_vs_ktc"] is None
    assert out[2]["ktc_rank"] == 2 and out[2]["delta_vs_ktc"] == -1


def test_patch_board_ktc_clears_stale_ranks():
    # a player Dynasty Daddy dropped must lose its old rank, not keep it
    board = [{"name": "Old Guy", "ktc_rank": 5, "delta_vs_ktc": 4}]
    out = patch_board_ktc(board, {})
    assert out[0]["ktc_rank"] is None and out[0]["delta_vs_ktc"] is None
