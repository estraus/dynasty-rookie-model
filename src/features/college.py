"""Shared college per-player-season table with market shares.

Pivots the long CFBD player/team stats to one row per (player_id, season, team),
computes team-relative market shares, and z-scores those shares within season
(league-wide pass rates drift, so raw shares aren't comparable across eras).

Every position builder consumes this. Point-in-time safe: college data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.util import ROOT

# share columns that get within-season normalization
SHARE_COLS = [
    "dominator", "rec_yd_share", "rec_td_share", "yptpa", "reception_share",
    "scrimmage_share", "td_share",
]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


CAT_PREFIX = {"passing": "pass", "rushing": "rush", "receiving": "rec"}


def _player_wide() -> pd.DataFrame:
    cs = pd.read_parquet(ROOT / "data/raw/cfbd_player_season.parquet")
    cs = cs[cs.category.isin(CAT_PREFIX)].copy()
    cs["key"] = cs.category.map(CAT_PREFIX) + "_" + cs.stat_type  # pass_YDS, rec_TD, ...
    cs["stat"] = _num(cs["stat"])
    wide = cs.pivot_table(
        index=["player_id", "player", "season", "team", "conference"],
        columns="key", values="stat", aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def _team_wide() -> pd.DataFrame:
    ts = pd.read_parquet(ROOT / "data/raw/cfbd_team_season.parquet")
    ts["stat_value"] = _num(ts["stat_value"])
    keep = ["netPassingYards", "passingTDs", "passAttempts", "passCompletions",
            "rushingYards", "rushingTDs", "games"]
    ts = ts[ts.stat_name.isin(keep)]
    wide = ts.pivot_table(index=["team", "season"], columns="stat_name",
                          values="stat_value", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide.rename(columns={
        "netPassingYards": "team_pass_yds", "passingTDs": "team_pass_td",
        "passAttempts": "team_pass_att", "passCompletions": "team_comp",
        "rushingYards": "team_rush_yds", "rushingTDs": "team_rush_td",
        "games": "team_games",
    })


def _safe_div(a, b):
    return np.where((b.notna()) & (b > 0), a / b, np.nan)


def build() -> pd.DataFrame:
    """One row per (player_id, season, team) with raw stats, shares, and z-shares."""
    p = _player_wide()
    t = _team_wide()
    # standardize player stat column names (missing category -> NaN -> 0 for the player)
    ren = {"pass_YDS": "pass_yds", "pass_TD": "pass_td", "pass_ATT": "pass_att",
           "pass_INT": "pass_int", "pass_COMPLETIONS": "pass_comp", "pass_YPA": "pass_ypa",
           "rush_YDS": "rush_yds", "rush_TD": "rush_td", "rush_CAR": "rush_car",
           "rush_YPC": "rush_ypc",
           "rec_YDS": "rec_yds", "rec_TD": "rec_td", "rec_REC": "rec", "rec_YPR": "rec_ypr"}
    p = p.rename(columns={k: v for k, v in ren.items() if k in p.columns})
    for c in ren.values():
        if c not in p.columns:
            p[c] = np.nan

    df = p.merge(t, on=["team", "season"], how="left")
    stat_cols = ["pass_yds", "pass_td", "pass_att", "pass_int", "pass_comp", "rush_yds",
                 "rush_td", "rush_car", "rec_yds", "rec_td", "rec"]
    z = df[stat_cols].fillna(0)  # a player with no receiving simply has 0 rec, etc.

    # market shares (receiving denominators come from team passing totals)
    df["dominator"] = 0.5 * _safe_div(z.rec_yds, df.team_pass_yds) + \
        0.5 * _safe_div(z.rec_td, df.team_pass_td)
    df["rec_yd_share"] = _safe_div(z.rec_yds, df.team_pass_yds)
    df["rec_td_share"] = _safe_div(z.rec_td, df.team_pass_td)
    df["yptpa"] = _safe_div(z.rec_yds, df.team_pass_att)
    df["reception_share"] = _safe_div(z.rec, df.team_comp)
    df["scrimmage_share"] = _safe_div(z.rush_yds + z.rec_yds, df.team_rush_yds + df.team_pass_yds)
    df["td_share"] = _safe_div(z.rush_td + z.rec_td, df.team_rush_td + df.team_pass_td)

    # QB passing efficiency, era-normalized among real passers (>=100 attempts)
    df["pass_td_rate"] = _safe_div(z.pass_td, z.pass_att)
    df["_passer"] = z.pass_att >= 100
    for col in ["pass_ypa", "pass_td_rate"]:
        active = df[df._passer & df[col].notna()]
        stats = active.groupby("season")[col].agg(["mean", "std"])
        df = df.merge(stats.rename(columns={"mean": f"_m_{col}", "std": f"_s_{col}"}),
                      on="season", how="left")
        df[f"z_{col}"] = _safe_div(df[col] - df[f"_m_{col}"], df[f"_s_{col}"])
        df = df.drop(columns=[f"_m_{col}", f"_s_{col}"])

    # within-season z-scores over players with real receiving/rushing usage
    for col in SHARE_COLS:
        active = df[df[col].notna() & (df[col] > 0)]
        stats = active.groupby("season")[col].agg(["mean", "std"])
        df = df.merge(stats.rename(columns={"mean": f"_m_{col}", "std": f"_s_{col}"}),
                      on="season", how="left")
        df[f"z_{col}"] = _safe_div(df[col] - df[f"_m_{col}"], df[f"_s_{col}"])
        df = df.drop(columns=[f"_m_{col}", f"_s_{col}"])
    return df.drop(columns=["_passer"])


def age_at_season(season: int, draft_age: float, draft_year: int) -> float:
    """Approx age during college season `season` from age at draft.
    A player's final college fall precedes the spring draft, so age in season S
    is roughly draft_age minus the years between S's fall and draft spring."""
    if pd.isna(draft_age):
        return np.nan
    return float(draft_age) - (draft_year - 1 - season)


def player_seasons(college: pd.DataFrame, cfbd_player_id: str) -> pd.DataFrame:
    """All college seasons for one player, sorted oldest->newest."""
    return college[college.player_id == str(cfbd_player_id)].sort_values("season")
