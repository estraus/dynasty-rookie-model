"""TE features: age-adjusted receiving share. Capital is expected to dominate here."""
from __future__ import annotations

import numpy as np

from src.features.college import age_at_season


def features(seasons, ctx) -> dict:
    if seasons.empty:
        return {}
    final = seasons.iloc[-1]
    age = age_at_season(int(final.season), ctx["draft_age"], ctx["draft_year"])
    # age-adjust: same production is more impressive from a younger TE
    ageadj = final.rec_yd_share / age if (age and not np.isnan(age) and age > 0) else np.nan
    return {
        "te_rec_yd_share": final.rec_yd_share,
        "te_z_rec_yd_share": final.z_rec_yd_share,
        "te_rec_share_ageadj": ageadj,
    }
