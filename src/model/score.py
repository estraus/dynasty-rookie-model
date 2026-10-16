"""Score the 2026 rookie class and export board.json + board.csv.

Pipeline (frozen model, no re-tuning):
  1. pull the score-class draft picks (cached)
  2. crosswalk them to CFBD college ids
  3. assemble the same features used in training
  4. expected 3-season VOR (src.model.value) x position horizon multiplier ->
     historical percentile = score 1-100 (superflex enters via QB replacement)
  5. the full residual model gives the 'lean vs capital' and top-3 drivers
  6. tiers from score bands; KTC delta if overrides/ktc_<class>.csv exists

Run:  python -m src.model.score
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.features import college
from src.features.build import assemble, combine_medians
from src.ids import crosswalk
from src.util import ROOT, config

PROC = ROOT / "data/processed"
DRIVER_LABELS = {
    "log_pick": "draft capital", "age": "age", "early_declare": "early declare",
    "power_conf": "power conference", "speed_score": "speed score",
    "combine_missing": "no combine",
    "wr_z_dominator": "dominator", "wr_best_dominator": "peak dominator",
    "wr_z_yptpa": "yards / team pass att", "wr_breakout_age": "breakout age",
    "wr_final_vs_best_dom": "final-vs-best dominator",
    "rb_z_scrimmage_share": "scrimmage share", "rb_z_reception_share": "reception share",
    "rb_z_td_share": "TD share",
    "te_z_rec_yd_share": "receiving share", "te_rec_share_ageadj": "age-adj rec share",
    "qb_rush_ypg": "rushing / game", "qb_rush_tdpg": "rush TD / game",
    "qb_z_ypa": "yards / attempt", "qb_z_td_rate": "TD rate",
    "qb_career_pass_att": "career attempts", "qb_career_games": "career games",
}


from src.ingest.ktc import _draft_class  # noqa: E402 - shared cached pull


def _score_members() -> pd.DataFrame:
    """Crosswalk + prep the score-class skill players for feature assembly."""
    cfg = config()
    d = _draft_class()
    d = d[d.position.isin(cfg["positions"])].copy()
    d["norm_name"] = d.pfr_player_name.map(crosswalk.norm_name)
    d["norm_school"] = d.college.map(crosswalk.norm_school)
    matched = crosswalk.match_draftees(d)
    m = matched.merge(d[["gsis_id", "age"]], on="gsis_id", how="left")
    m = m[m.gsis_id.notna()].drop_duplicates(subset="gsis_id")
    return m.rename(columns={"pfr_player_name": "player", "season": "draft_season"})


def _value_scores(feats: pd.DataFrame, vfrozen: dict, bfrozen: dict) -> pd.DataFrame:
    """Expected value -> 6-season dynasty value -> historical percentile (1-100).

    value (exp_vor)  expected ppg over replacement, mean of best 2 of first 3
                     seasons, from the shipped value model (capital everywhere
                     unless a challenger cleared the bootstrap bar)
    dyn_value        value x position horizon multiplier (years 4-6 / 1-3), so
                     front-loaded RBs and back-loaded WR/TEs compare fairly
    score_1_100      percentile of dyn_value among every training player's
                     fitted dyn_value -- an 80 means what it meant historically
    lean             the full production/age residual model's adjustment, in
                     ppg VOR; lean_validated says whether it beat capital
    value_range      80% range of realized value for players predicted alike
    p_hit            binary capital model (kept for continuity)
    """
    from src.model.value import interval_for

    hist = np.sort(np.concatenate([vfrozen[p]["hist_value"] for p in vfrozen]))
    out = feats.copy()
    cols = {k: [] for k in ("exp_vor", "dyn_value", "lean", "range_lo", "range_hi", "p_hit")}
    for i in out.index:
        row = out.loc[[i]]
        pos = row.position.iloc[0]
        vf, bf = vfrozen[pos], bfrozen[pos]
        val = float(vf["models"][vf["chosen"]].predict(row)[0])
        lo, hi = interval_for(vf["intervals"], val)
        cols["exp_vor"].append(round(val, 2))
        cols["dyn_value"].append(round(val * vf["horizon"], 2))
        cols["lean"].append(round(float(vf["models"][vf["lean_from"]].adjustment(row)[0]), 3))
        cols["range_lo"].append(round(lo, 2)); cols["range_hi"].append(round(hi, 2))
        cols["p_hit"].append(float(bf["model"].predict_proba(
            row[bf["cols"]].astype(float).fillna(0).values)[:, 1][0]))
    for k, v in cols.items():
        out[k] = v
    pct = np.searchsorted(hist, out.dyn_value.values, side="right") / len(hist)
    out["score_1_100"] = np.clip(np.round(pct * 100).astype(int), 1, 100)
    out["lean_validated"] = out.position.map(lambda p: bool(vfrozen[p]["lean_validated"]))
    out["reliability"] = out.position.map(lambda p: vfrozen[p]["reliability"])
    return out


def _tier(score: int) -> int:
    for band in config()["board"]["tiers"]:
        if score >= band["min"]:
            return band["tier"]
    return config()["board"]["tiers"][-1]["tier"]


def _drivers(feats: pd.DataFrame, vfrozen: dict) -> list:
    """Top-3 drivers per player, in ppg-VOR units.

    'draft capital' = the isotonic capital value minus the position's average
    capital value (how much the slot itself is worth); the rest are the full
    residual model's per-feature contributions (standardized coef x z).
    """
    all_drivers = []
    for i in feats.index:
        row = feats.loc[[i]]
        pos = row.position.iloc[0]
        vf = vfrozen[pos]
        full = vf["models"]["full"]
        cap = float(full.capital(row)[0]) - vf["cap_mean"]
        contrib = full.contributions(row)[0]
        sc = full.ridge.named_steps["standardscaler"]
        x = row[full.cols].astype(float).fillna(0.0).values[0]
        z = (x - sc.mean_) / sc.scale_
        items = [("draft capital", cap, float(row.pick.iloc[0]), None)]
        items += [(DRIVER_LABELS.get(c, c), float(contrib[j]),
                   None if pd.isna(row[c].iloc[0]) else round(float(x[j]), 2),
                   round(float(z[j]), 2)) for j, c in enumerate(full.cols)]
        items.sort(key=lambda t: -abs(t[1]))
        all_drivers.append([
            {"name": n, "direction": "positive" if v >= 0 else "negative",
             "magnitude": round(abs(v), 3), "value": val, "z": zz}
            for n, v, val, zz in items[:3]])
    return all_drivers


def _comps(feats: pd.DataFrame, frozen: dict, n: int = 5) -> list:
    """Top-n historical comps per player: nearest neighbors in the full-LR
    feature space with dimensions weighted by |coef| -- similar where the model
    cares. why = the weightiest features where the two players nearly match."""
    hist = pd.read_parquet(ROOT / "data/interim/features.parquet")
    hist = hist[hist.label_complete.astype(bool)].copy()
    peaks = pd.read_parquet(ROOT / "data/interim/labels.parquet")
    hist = hist.merge(peaks[["gsis_id", "best_pos_rank"]], on="gsis_id", how="left")

    all_comps = []
    for r in feats.itertuples():
        fp = frozen[r.position]
        pipe = fp["full_lr"]
        scaler = pipe.named_steps["standardscaler"]
        cols = fp["full_cols"]
        w = np.abs(pipe.named_steps["logisticregression"].coef_[0])
        w = w / w.sum()
        pool = hist[hist.position == r.position].reset_index(drop=True)
        Xp = (pool[cols].astype(float).fillna(0).values - scaler.mean_) / scaler.scale_
        x = (feats.loc[[r.Index], cols].astype(float).fillna(0).values[0]
             - scaler.mean_) / scaler.scale_
        dist = np.sqrt(((Xp - x) ** 2 * w).sum(axis=1))
        comps = []
        for i in np.argsort(dist)[:n]:
            diffs = np.abs(Xp[i] - x)
            # a feature explains the comp if it matters (weight) and matches (diff)
            match = w * (1 - np.minimum(diffs, 2.0) / 2.0)
            why = [DRIVER_LABELS.get(cols[j], cols[j]) for j in np.argsort(-match)[:2]]
            row = pool.iloc[i]
            comps.append({
                "name": row.player, "year": int(row.draft_season),
                "pick": int(row.pick), "hit": bool(row.hit),
                "peak": None if pd.isna(row.best_pos_rank) else int(row.best_pos_rank),
                "why": why,
            })
        all_comps.append(comps)
    return all_comps


def _ktc(feats: pd.DataFrame) -> pd.DataFrame:
    """Market ranks from overrides/ktc_<class>.csv: ktc_rank drives the delta
    view; any other <src>_rank columns ride along as market_ranks context."""
    yr = config()["seasons"]["score_class"]
    path = ROOT / "overrides" / f"ktc_{yr}.csv"
    feats = feats.copy()
    if not path.exists():
        feats["ktc_rank"] = pd.NA
        feats["delta_vs_ktc"] = pd.NA
        feats["market_ranks"] = [{} for _ in range(len(feats))]
        return feats
    ktc = pd.read_csv(path)
    rank_cols = [c for c in ktc.columns if c.endswith("_rank")]
    ktc["_k"] = ktc.name.map(crosswalk.norm_name)
    feats["_k"] = feats.player.map(crosswalk.norm_name)
    feats = feats.merge(ktc[["_k", *rank_cols]], on="_k", how="left").drop(columns="_k")
    feats["model_rank"] = feats.score_1_100.rank(ascending=False, method="min")
    feats["delta_vs_ktc"] = feats.ktc_rank - feats.model_rank  # +ve = model higher than KTC
    others = [c for c in rank_cols if c != "ktc_rank"]
    feats["market_ranks"] = [
        {c[:-5]: int(v) for c in others if pd.notna(v := row[c])}
        for _, row in feats[others].iterrows()]
    return feats


def _college_line(feats: pd.DataFrame, col: pd.DataFrame) -> dict:
    """Compact final-season college stat line per player for the detail drawer."""
    lines = {}
    for r in feats.itertuples():
        if pd.isna(r.cfbd_player_id):
            lines[r.Index] = {}
            continue
        s = college.player_seasons(col, r.cfbd_player_id)
        if s.empty:
            lines[r.Index] = {}
            continue
        f = s.iloc[-1]
        lines[r.Index] = {k: (None if pd.isna(f[k]) else round(float(f[k]), 1)) for k in
                          ["rec", "rec_yds", "rec_td", "rush_yds", "rush_td",
                           "pass_yds", "pass_td"] if k in s.columns}
    return lines


COMBINE_COVERAGE_FLOOR = 0.25  # recent classes test ~50-70%; far below = broken pull


def _check_combine_coverage(feats: pd.DataFrame) -> None:
    """Fail loudly if the score class has (almost) no combine data, which means
    the combine pull missed this class rather than that players skipped it."""
    if feats.age.isna().any():
        raise RuntimeError(f"missing age for {feats.loc[feats.age.isna(), 'player'].tolist()}; "
                           "a NaN age is read as 0 years old downstream")
    tested = 1 - feats.combine_missing.mean()
    yr = config()["seasons"]["score_class"]
    print(f"combine coverage for {yr}: {tested:.0%} tested")
    if tested < COMBINE_COVERAGE_FLOOR:
        raise RuntimeError(
            f"only {tested:.0%} of the {yr} class has combine data; refresh it with "
            "`python -m src.ingest.nflverse` before scoring")


def build_board() -> pd.DataFrame:
    frozen = joblib.load(PROC / "models.joblib")          # binary: p_hit + comps
    vfrozen = joblib.load(PROC / "value_models.joblib")   # value: score, lean, drivers
    col = college.build()
    members = _score_members()
    # fill combine gaps with TRAINING medians (what the model learned on)
    train_feats = pd.read_parquet(ROOT / "data/interim/features.parquet")
    feats = assemble(members, col, medians=combine_medians(train_feats)).reset_index(drop=True)
    _check_combine_coverage(feats)
    feats = feats.merge(
        members[["gsis_id", "cfbd_player_id", "match_method", "college"]],
        on="gsis_id", how="left")

    print("horizon multipliers:", {p: vfrozen[p]["horizon"] for p in vfrozen})
    feats = _value_scores(feats, vfrozen, frozen)
    feats["tier"] = feats.score_1_100.map(_tier)
    # attach per-player drivers + college line as columns (index-aligned pre-sort)
    feats["drivers"] = _drivers(feats, vfrozen)
    feats["comps"] = _comps(feats, frozen)
    lines = _college_line(feats, col)
    feats["college_stats"] = [lines.get(i, {}) for i in feats.index]
    feats = _ktc(feats)
    feats = feats.merge(_draft_class()[["gsis_id", "round"]].drop_duplicates("gsis_id"),
                        on="gsis_id", how="left")

    # score ties (isotonic plateaus) break on dynasty value, then draft slot
    feats = feats.sort_values(["score_1_100", "dyn_value", "pick"],
                              ascending=[False, False, True], kind="stable").reset_index(drop=True)
    return pd.DataFrame({
        "player_id": feats.gsis_id, "name": feats.player, "pos": feats.position,
        "school": feats.college, "pick": feats.pick.astype(int), "round": feats["round"],
        "age": feats.age, "score_1_100": feats.score_1_100, "p_hit": feats.p_hit.round(4),
        "exp_vor": feats.exp_vor, "dyn_value": feats.dyn_value,
        "value_range": [[lo, hi] for lo, hi in zip(feats.range_lo, feats.range_hi)],
        "reliability": feats.reliability,
        "lean": feats.lean, "lean_validated": feats.lean_validated,
        "tier": feats.tier, "ktc_rank": feats.ktc_rank,
        "delta_vs_ktc": feats.delta_vs_ktc, "market_ranks": feats.market_ranks,
        "drivers": feats.drivers, "comps": feats.comps,
        "college_stats": feats.college_stats,
    })


def main() -> None:
    board = build_board()
    PROC.mkdir(parents=True, exist_ok=True)
    # via pandas to_json so NaN -> null (browsers reject bare NaN; Python accepts it)
    records = json.loads(board.to_json(orient="records"))
    payload = json.dumps(records, indent=2)
    (PROC / "board.json").write_text(payload)
    flat = board.drop(columns=["drivers", "college_stats", "comps", "market_ranks", "value_range"])
    flat.to_csv(PROC / "board.csv", index=False)
    # app reads from app/public at build time
    app_data = ROOT / "app" / "public"
    if app_data.exists():
        (app_data / "board.json").write_text(json.dumps(records))

    yr = config()["seasons"]["score_class"]
    print(f"scored {len(board)} players in the {yr} class")
    print(board.head(20)[["name", "pos", "pick", "score_1_100", "exp_vor", "dyn_value", "lean", "tier"]]
          .to_string(index=False))
    has_college = board.college_stats.map(bool).mean()
    print(f"\ncollege stat line present: {has_college:.0%} | tiers: "
          f"{board.tier.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
