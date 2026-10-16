"""QB features: rushing per game, era-normalized YPA & TD rate, career volume.

Superflex premium is NOT baked in here -- it's applied at the cross-position
board layer. CFBD season stats lack player game counts, so per-game rushing uses
team games (player ~ team games as a full-season starter) as the denominator.
"""
from __future__ import annotations

import numpy as np


def features(seasons, ctx) -> dict:
    if seasons.empty:
        return {}
    final = seasons.iloc[-1]
    games = seasons.team_games.fillna(0).sum()
    rush_yds = seasons.rush_yds.fillna(0).sum()
    rush_td = seasons.rush_td.fillna(0).sum()
    return {
        "qb_rush_ypg": rush_yds / games if games > 0 else np.nan,
        "qb_rush_tdpg": rush_td / games if games > 0 else np.nan,
        "qb_z_ypa": final.z_pass_ypa,
        "qb_z_td_rate": final.z_pass_td_rate,
        "qb_career_pass_att": seasons.pass_att.fillna(0).sum(),
        "qb_career_games": games,
    }
