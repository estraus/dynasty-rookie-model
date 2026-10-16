"""Feature-set definitions shared by train.py and score.py (single source)."""

SHARED = ["log_pick", "age", "early_declare", "power_conf", "speed_score", "combine_missing"]

POSITION_FEATURES = {
    "WR": ["wr_z_dominator", "wr_best_dominator", "wr_z_yptpa", "wr_breakout_age",
           "wr_final_vs_best_dom"],
    "RB": ["rb_z_scrimmage_share", "rb_z_reception_share", "rb_z_td_share"],
    "TE": ["te_z_rec_yd_share", "te_rec_share_ageadj"],
    "QB": ["qb_rush_ypg", "qb_rush_tdpg", "qb_z_ypa", "qb_z_td_rate", "qb_career_games"],
}

BASELINE = ["log_pick"]  # capital-only: the bar the full model must clear


def full_features(pos: str) -> list:
    return SHARED + POSITION_FEATURES[pos]
