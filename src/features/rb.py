"""RB features: scrimmage-yard share, reception share (half-PPR), TD share.

Speed score (weight & 40 time) is added at the combine layer in build.py.
"""
from __future__ import annotations


def features(seasons, ctx) -> dict:
    if seasons.empty:
        return {}
    final = seasons.iloc[-1]
    return {
        "rb_scrimmage_share": final.scrimmage_share,
        "rb_z_scrimmage_share": final.z_scrimmage_share,
        "rb_reception_share": final.reception_share,
        "rb_z_reception_share": final.z_reception_share,
        "rb_td_share": final.td_share,
        "rb_z_td_share": final.z_td_share,
    }
