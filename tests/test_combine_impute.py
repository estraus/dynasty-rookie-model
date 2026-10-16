"""Regression: score-class combine gaps must not become 0 downstream.

The 2026 combine was never pulled, so every 2026 player had NaN speed score.
The per-class median was also NaN, the scorer's fillna(0) turned that into a
speed score of 0 (~10 SDs low), and it dominated lean, drivers and comps.
Hermetic: synthetic frames only, no caches needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import combine_medians, impute_combine


def _frame(rows):
    return pd.DataFrame(rows, columns=["position", "speed_score", "wt", "forty"])


TRAIN = impute_combine(_frame([
    ("RB", 100.0, 210.0, 4.50),
    ("RB", 110.0, 220.0, 4.45),
    ("RB", np.nan, np.nan, np.nan),      # opt-out: must not pull the median
    ("WR", 95.0, 195.0, 4.48),
    ("WR", 105.0, 200.0, 4.40),
]))


def test_training_behaviour_unchanged():
    """No medians passed = own position medians, as before the fix."""
    rb = TRAIN[TRAIN.position == "RB"]
    assert rb.combine_missing.tolist() == [0, 0, 1]
    assert rb.speed_score.iloc[2] == 105.0


def test_medians_use_only_tested_players():
    m = combine_medians(TRAIN)
    assert m["RB"]["speed_score"] == 105.0
    assert m["WR"]["forty"] == pytest.approx(4.44)


def test_all_missing_class_gets_training_medians():
    """The 2026 failure mode: a whole class with no combine rows."""
    score = _frame([("RB", np.nan, np.nan, np.nan), ("WR", np.nan, np.nan, np.nan)])
    out = impute_combine(score, combine_medians(TRAIN))
    assert out.combine_missing.tolist() == [1, 1]
    assert out.speed_score.tolist() == [105.0, 100.0]
    assert not out[["speed_score", "wt", "forty"]].isna().any().any()


def test_tested_players_keep_their_own_numbers():
    score = _frame([("RB", 120.0, 225.0, 4.40), ("RB", np.nan, np.nan, np.nan)])
    out = impute_combine(score, combine_medians(TRAIN))
    assert out.speed_score.tolist() == [120.0, 105.0]
    assert out.combine_missing.tolist() == [0, 1]


def test_unfillable_gap_raises_instead_of_zero():
    with pytest.raises(ValueError, match="TE"):
        impute_combine(_frame([("TE", np.nan, np.nan, np.nan)]), combine_medians(TRAIN))
