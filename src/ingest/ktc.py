"""KTC sanity ranks via the Dynasty Daddy public API (aggregates KeepTradeCut).

Not a scraper -- a stable JSON endpoint. Pulls today's superflex trade values,
keeps the score-class rookies (matched by name to the drafted class), ranks them
1..N by superflex value, and writes overrides/ktc_<year>.csv (name, ktc_rank)
which src.model.score reads for the board's delta view.

Run:  python -m src.ingest.ktc        # refresh KTC ranks, then re-run score
"""
from __future__ import annotations

import requests

from src.ids.crosswalk import norm_name
from src.util import ROOT, cached, config

API = "https://dynasty-daddy.com/api/v1/player/all/today"

# Dynasty Daddy market ids (enum verified against their bundle). Shared with
# src.ingest.sleeper; lives here because sleeper imports from this module.
DD_SOURCES = {"ktc": 0, "dp": 2, "ds": 3, "adp": 6, "dd": 14}


def _draft_class():
    """Score-class draft picks (cached). Lives here, not in score.py, so the
    CI refresh path never imports sklearn/joblib."""
    yr = config()["seasons"]["score_class"]

    def pull():
        import nfl_data_py as nfl
        return nfl.import_draft_picks([yr])

    return cached(f"nfl_draft_{yr}", pull)


def fetch_sf_values(market: int = 0) -> dict:
    """{normalized name: sf_trade_value} for all players Dynasty Daddy tracks.

    market ids from Dynasty Daddy's own enum (verified against their bundle):
    0=KeepTradeCut 1=FantasyCalc 2=DynastyProcess 3=DynastySuperflex
    6=DynastyDaddyADP 14=DynastyDaddy. Default = KTC.
    """
    r = requests.get(API, params={"market": market}, timeout=30)
    r.raise_for_status()
    out = {}
    for p in r.json():
        v = p.get("sf_trade_value")
        if v:
            out[norm_name(p["full_name"])] = v
    return out


def build() -> None:
    cfg = config()
    yr = cfg["seasons"]["score_class"]

    d = _draft_class()
    d = d[d.position.isin(cfg["positions"])].copy()
    d["nn"] = d.pfr_player_name.map(norm_name)
    # rank the class 1..N within each market that prices its players
    rank_cols = []
    for src, mid in DD_SOURCES.items():
        sv = d.nn.map(fetch_sf_values(market=mid))
        if src == "ktc":
            d["sf_value"] = sv
        col = f"{src}_rank"
        d[col] = sv.rank(ascending=False, method="min").astype("Int64")
        rank_cols.append(col)

    matched = d[d[rank_cols].notna().any(axis=1)].sort_values("ktc_rank")
    path = ROOT / "overrides" / f"ktc_{yr}.csv"
    matched[["pfr_player_name", "sf_value", *rank_cols]].rename(
        columns={"pfr_player_name": "name"}).to_csv(path, index=False)
    print(f"wrote {len(matched)}/{len(d)} {yr} rookies with market ranks {rank_cols} -> {path}")
    print("  (unmatched are players Dynasty Daddy doesn't track yet)")


if __name__ == "__main__":
    build()
