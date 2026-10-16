"""Labels: per-player-season fantasy PPG with league scoring, then hit flags.

League: superflex (positional premium applied later at the board layer, NOT here),
half-PPR, 6pt passing TD -- all coefficients live in config.yaml.

hit = a top-N positional PPG season (min games) within a player's first 3 NFL
seasons: top-12 QB, top-24 RB, top-24 WR, top-12 TE.

The ranking pool is league-wide (every player at that position that season), not
just rookies. Regular season only -- dynasty relevance is scored on REG.

Outputs:
  data/interim/season_ppg.parquet  every player-season: games, ppg, pos_rank, is_topN
  data/interim/labels.parquet      one row per drafted skill player 2013-2023

Run:  python -m src.labels.build_labels
"""
from __future__ import annotations

import pandas as pd

from src.util import ROOT, config

# weekly component -> config scoring key
SCORING = {
    "passing_yards": "passing_yards", "passing_tds": "passing_td",
    "interceptions": "interception", "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_td", "receptions": "reception",
    "receiving_yards": "receiving_yards", "receiving_tds": "receiving_td",
}
FUMBLE_COLS = ["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]
TWOPT_COLS = ["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"]


def score_weekly(w: pd.DataFrame, scoring: dict | None = None) -> pd.Series:
    """Fantasy points per player-week. Default = league scoring (config);
    pass a scoring dict to score under a different convention."""
    s = scoring or config()["league"]
    pts = sum(w[col].fillna(0) * s[key] for col, key in SCORING.items())
    pts = pts + w[FUMBLE_COLS].fillna(0).sum(axis=1) * s["fumble_lost"]
    pts = pts + w[TWOPT_COLS].fillna(0).sum(axis=1) * s["two_point"]
    # TE reception premium (Sleeper bonus_rec_te); market conventions pass 0
    tep = s.get("te_reception_bonus", 0)
    if tep:
        pts = pts + w["receptions"].fillna(0) * tep * (w["position"] == "TE")
    return pts


def season_ppg() -> pd.DataFrame:
    w = pd.read_parquet(ROOT / "data/raw/nfl_weekly.parquet")
    w = w[(w.season_type == "REG") & w.position.notna()].copy()
    w["pts"] = score_weekly(w)
    # a player's position for a season = their most frequent listed position
    pos = (w.groupby(["player_id", "season"])["position"]
             .agg(lambda s: s.value_counts().index[0]).rename("position"))
    agg = w.groupby(["player_id", "season"]).agg(
        player=("player_display_name", "first"),
        games=("week", "nunique"),
        points=("pts", "sum"),
    ).join(pos).reset_index()
    agg["ppg"] = agg.points / agg.games

    cfg = config()
    min_g = cfg["labels"]["min_games"]
    ranks = cfg["labels"]["hit_rank"]
    # rank within (season, position) among qualified players (games >= min)
    agg["pos_rank"] = pd.NA
    agg["is_topN"] = False
    for pos_name, topn in ranks.items():
        mask = (agg.position == pos_name) & (agg.games >= min_g)
        r = agg.loc[mask].groupby("season")["ppg"].rank(ascending=False, method="min")
        agg.loc[mask, "pos_rank"] = r
        agg.loc[mask, "is_topN"] = r <= topn
    return agg


def build_labels(ppg: pd.DataFrame) -> pd.DataFrame:
    cfg = config()
    lo, hi = cfg["seasons"]["train_classes"]
    window = cfg["labels"]["window_seasons"]
    max_season = int(ppg.season.max())

    d = pd.read_parquet(ROOT / "data/raw/nfl_draft.parquet")
    d = d[(d.season.between(lo, hi)) & (d.position.isin(cfg["positions"]))].copy()

    # topN seasons keyed by (player_id, season) for fast lookup
    topn = ppg.set_index(["player_id", "season"])

    rows = []
    for r in d.itertuples():
        rookie = r.season
        win = list(range(rookie, rookie + window))
        observed = [s for s in win if s <= max_season]
        seasons = [(s, topn.loc[(r.gsis_id, s)]) for s in observed
                   if (r.gsis_id, s) in topn.index]
        hit_seasons = [s for s, row in seasons if bool(row.is_topN)]
        best_rank = min((row.pos_rank for _, row in seasons
                         if pd.notna(row.pos_rank)), default=pd.NA)
        rows.append({
            "gsis_id": r.gsis_id, "player": r.pfr_player_name, "position": r.position,
            "draft_season": rookie, "round": r.round, "pick": r.pick,
            "seasons_observed": len(observed),
            "label_complete": len(observed) == window,
            "hit": len(hit_seasons) > 0,
            "hit_season": min(hit_seasons) if hit_seasons else pd.NA,
            "best_pos_rank": best_rank,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ppg = season_ppg()
    (ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
    ppg.to_parquet(ROOT / "data/interim/season_ppg.parquet", index=False)

    labels = build_labels(ppg)
    labels.to_parquet(ROOT / "data/interim/labels.parquet", index=False)

    comp = labels[labels.label_complete]
    print(f"season_ppg: {len(ppg):,} player-seasons")
    print(f"labels: {len(labels)} drafted players, {len(comp)} with complete 3yr window")
    print("hit rate (complete windows only) by position:")
    for pos in config()["positions"]:
        p = comp[comp.position == pos]
        print(f"  {pos}: {p.hit.sum()}/{len(p)} hits ({100*p.hit.mean():.1f}%)")
    print(f"incomplete-window classes: "
          f"{sorted(labels[~labels.label_complete].draft_season.unique().tolist())}")


if __name__ == "__main__":
    main()
