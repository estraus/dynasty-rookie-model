"""Assemble the model feature matrix: shared + combine + per-position + labels.

One row per drafted skill player (2013-2023) that we have a label for. Players
without CFBD college data (crosswalk misses) still get a row -- their college
features are NaN and the combine/capital features carry them, with the model's
missingness handling picking up the slack.

Run:  python -m src.features.build
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import college, qb, rb, te, wr
from src.util import ROOT, config

POS_BUILDERS = {"WR": wr, "RB": rb, "TE": te, "QB": qb}
EARLY_DECLARE_MAX_COLLEGE_YEARS = 3


COMBINE_COLS = ["speed_score", "wt", "forty"]


def _combine() -> pd.DataFrame:
    # via the ingest helper (not the raw parquet) so a stale cache that predates
    # the score class is refreshed instead of silently yielding no combine data
    from src.ingest.nflverse import combine
    cb = combine()
    cb = cb[["pfr_id", "wt", "forty"]].dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    # speed score = (weight * 200) / (40 time ^ 4); opt-outs -> NaN, imputed later
    cb["speed_score"] = np.where(cb.forty > 0, cb.wt * 200 / cb.forty**4, np.nan)
    return cb


PASSTHROUGH = ["hit", "label_complete"]  # kept if present (training), absent when scoring


def combine_medians(feats: pd.DataFrame) -> dict:
    """{position: {col: median}} over players who actually tested.

    Computed on the training matrix and reused when scoring, so a score-class
    player with no combine gets the same fill the model was trained on.
    """
    tested = feats[feats.combine_missing == 0] if "combine_missing" in feats else feats
    meds = {pos: {c: float(g[c].median()) for c in COMBINE_COLS}
            for pos, g in tested.groupby("position")}
    if "age" in feats:  # fallback fill for score-class players with no birthdate
        for pos, g in feats.groupby("position"):
            meds.setdefault(pos, {})["age"] = float(g.age.median())
    return meds


def impute_combine(out: pd.DataFrame, medians: dict | None = None) -> pd.DataFrame:
    """Flag missing combine data, then fill it.

    Missingness is nonrandom (opt-outs skew athletic), so the flag carries
    signal and the fill is a neutral position median. With `medians` (training
    values) every gap uses them; without, the frame's own position medians are
    used. Never leaves NaN behind for these columns: a NaN here used to become
    0 downstream, i.e. a speed score ~10 SDs below the position norm.
    """
    out = out.copy()
    out["combine_missing"] = out.speed_score.isna().astype(int)
    for c in COMBINE_COLS:
        own = out.groupby("position")[c].transform("median")
        if medians is not None:
            train = out.position.map(lambda p: medians.get(p, {}).get(c, np.nan))
            fill = train.fillna(own)
        else:
            fill = own
        out[c] = out[c].fillna(fill)
        if out[c].isna().any():
            bad = sorted(out.loc[out[c].isna(), "position"].unique())
            raise ValueError(f"no {c} median available for {bad}; combine data missing?")
    return out


def age_on_sept1(birthdate: pd.Series, draft_season: pd.Series) -> pd.Series:
    """Whole years old on Sept 1 of the draft year (rookie-season start).

    Matches nflverse's own draft `age` for ~95% of players that have both.
    """
    bd = pd.to_datetime(birthdate, errors="coerce")
    ref = pd.to_datetime(draft_season.astype("Int64").astype(str) + "-09-01", errors="coerce")
    had_bday = (ref.dt.month > bd.dt.month) | (
        (ref.dt.month == bd.dt.month) & (ref.dt.day >= bd.dt.day))
    return (ref.dt.year - bd.dt.year - (~had_bday).astype(int)).astype(float).where(bd.notna())


def backfill_age(members: pd.DataFrame, ids: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill missing draft `age` from the nflverse id table's birthdate.

    nflverse leaves age blank for some late-round picks (12 of 80 skill players
    in 2026). Downstream fillna(0) then read them as 0-year-olds, ~15-25 SDs
    young, which dominated their drivers. Birthdate is known at draft time, so
    this is point-in-time safe.
    """
    if not members.age.isna().any():
        return members
    if ids is None:
        ids = pd.read_parquet(ROOT / "data/raw/nfl_ids.parquet")
    bd = ids[["gsis_id", "birthdate"]].dropna().drop_duplicates("gsis_id")
    out = members.merge(bd, on="gsis_id", how="left")
    out["age"] = out.age.fillna(age_on_sept1(out.birthdate, out.draft_season))
    return out.drop(columns="birthdate")


def assemble(members: pd.DataFrame, col: pd.DataFrame | None = None,
             medians: dict | None = None) -> pd.DataFrame:
    """Build the feature matrix for a set of drafted players.

    `members` needs: gsis_id, pfr_player_id, player, position, draft_season, pick,
    cfbd_player_id, age. Optional hit/label_complete are passed through. Works for
    both the training classes and the 2026 scoring class -- same features, no leak.
    """
    members = backfill_age(members)
    if medians is not None and members.age.isna().any():
        # last resort when scoring: the training position median, never 0
        members = members.copy()
        members["age"] = members.age.fillna(
            members.position.map(lambda p: medians.get(p, {}).get("age", np.nan)))
    cfg = config()
    col = college.build() if col is None else col
    combine = _combine()
    power = set(cfg["power_conferences"])

    rows = []
    for r in members.itertuples():
        ctx = {"draft_age": r.age, "draft_year": r.draft_season}
        feat = {
            "gsis_id": r.gsis_id, "pfr_player_id": r.pfr_player_id,
            "player": r.player, "position": r.position,
            "draft_season": r.draft_season, "pick": r.pick,
            "log_pick": np.log(r.pick), "age": r.age,
        }
        for c in PASSTHROUGH:
            if hasattr(r, c):
                feat[c] = getattr(r, c)
        seasons = (college.player_seasons(col, r.cfbd_player_id)
                   if pd.notna(r.cfbd_player_id) else col.iloc[0:0])
        if not seasons.empty:
            first_season = int(seasons.iloc[0].season)
            final = seasons.iloc[-1]
            feat["final_college_season"] = int(final.season)
            feat["college_years"] = len(seasons)
            feat["early_declare"] = int(r.draft_season - first_season <= EARLY_DECLARE_MAX_COLLEGE_YEARS)
            feat["power_conf"] = int(final.conference in power)
        else:
            feat["final_college_season"] = np.nan
            feat["college_years"] = np.nan
            feat["early_declare"] = int(r.age <= 21) if pd.notna(r.age) else 0
            feat["power_conf"] = 0
        feat.update(POS_BUILDERS[r.position].features(seasons, ctx))
        rows.append(feat)

    out = pd.DataFrame(rows).merge(
        combine[["pfr_id", "speed_score", "wt", "forty"]],
        left_on="pfr_player_id", right_on="pfr_id", how="left").drop(columns=["pfr_id"])

    return impute_combine(out, medians)


def build() -> pd.DataFrame:
    """Training feature matrix: labeled 2013-2023 classes."""
    labels = pd.read_parquet(ROOT / "data/interim/labels.parquet")
    xwalk = pd.read_parquet(ROOT / "data/interim/crosswalk.parquet")[
        ["gsis_id", "pfr_player_id", "cfbd_player_id"]]
    draft = pd.read_parquet(ROOT / "data/raw/nfl_draft.parquet")[["gsis_id", "age"]]
    # drop players with no gsis_id: never appeared in nflverse (busts) + NaN keys
    # cartesian-blow up merges.
    labels = labels[labels.gsis_id.notna()]
    members = labels.merge(xwalk, on="gsis_id", how="left").merge(draft, on="gsis_id", how="left")
    return assemble(members)


def main() -> None:
    out = build()
    (ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
    out.to_parquet(ROOT / "data/interim/features.parquet", index=False)
    print(f"features: {len(out)} players x {out.shape[1]} cols")
    print("per position (complete-window rows):")
    comp = out[out.label_complete]
    for pos in config()["positions"]:
        p = comp[comp.position == pos]
        ncol = p.drop(columns=["gsis_id", "player"]).notna().mean().mean()
        print(f"  {pos}: {len(p)} rows, combine_missing={p.combine_missing.mean():.0%}, "
              f"college_data={p.final_college_season.notna().mean():.0%}")


if __name__ == "__main__":
    main()
