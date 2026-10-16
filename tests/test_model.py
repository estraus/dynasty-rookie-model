"""Model checks: frozen artifacts load, produce valid probabilities, and the
capital model is monotonic (a better pick never scores lower)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.util import ROOT

MODELS = ROOT / "data/processed/models.joblib"


def _frozen():
    if not MODELS.exists():
        import pytest  # noqa: PLC0415
        pytest.skip("models not trained; run python -m src.model.train")
    return joblib.load(MODELS)


def test_all_positions_frozen():
    fr = _frozen()
    assert set(fr) == {"QB", "RB", "WR", "TE"}
    for pos, m in fr.items():
        assert m["kind"] in ("lr", "lgbm")
        assert len(m["train_phit"]) > 0
        assert (m["train_phit"] >= 0).all() and (m["train_phit"] <= 1).all()


def test_predictions_are_probabilities():
    fr = _frozen()
    f = pd.read_parquet(ROOT / "data/interim/features.parquet")
    for pos, m in fr.items():
        d = f[f.position == pos]
        X = d[m["cols"]].astype(float).fillna(0.0).values
        p = m["model"].predict_proba(X)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()


def test_better_pick_scores_higher():
    """Core invariant: at pick 1, P(hit) > P(hit) at pick 200 (capital signal)."""
    fr = _frozen()
    f = pd.read_parquet(ROOT / "data/interim/features.parquet")
    for pos, m in fr.items():
        d = f[f.position == pos].iloc[[0]].copy()
        cols = m["cols"]
        early, late = d[cols].copy(), d[cols].copy()
        early["log_pick"] = np.log(1)
        late["log_pick"] = np.log(200)
        pe = m["model"].predict_proba(early.astype(float).fillna(0).values)[:, 1][0]
        pl = m["model"].predict_proba(late.astype(float).fillna(0).values)[:, 1][0]
        assert pe > pl, f"{pos}: pick-1 P {pe:.3f} !> pick-200 P {pl:.3f}"


