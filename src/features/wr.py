"""WR features: dominator, breakout age, yards per team pass attempt, final-vs-best.

`seasons` is one player's college seasons (oldest->newest) from college.build().
`ctx` carries draft_age / draft_year for age math. Uses final-team season for
point-in-time production, career best for breakout/decline signals.
"""
from __future__ import annotations

import numpy as np

from src.features.college import age_at_season

BREAKOUT_DOM = 0.20  # first season at 20%+ dominator = breakout


def features(seasons, ctx) -> dict:
    if seasons.empty:
        return {}
    final = seasons.iloc[-1]
    best_dom = seasons.dominator.max()
    broke = seasons[seasons.dominator >= BREAKOUT_DOM]
    breakout_age = (age_at_season(int(broke.iloc[0].season), ctx["draft_age"], ctx["draft_year"])
                    if not broke.empty else np.nan)
    return {
        "wr_dominator": final.dominator,
        "wr_z_dominator": final.z_dominator,
        "wr_best_dominator": best_dom,
        "wr_z_yptpa": final.z_yptpa,
        "wr_breakout_age": breakout_age,
        # decline signal: negative means final season fell off the career peak
        "wr_final_vs_best_dom": final.dominator - best_dom,
    }
