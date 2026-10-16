"""Score the 2026 rookie class and export board.json + board.csv.

Pipeline (frozen model, no re-tuning):
  1. pull the score-class draft picks (cached)
  2. crosswalk them to CFBD college ids
  3. assemble the same features used in training
  4. capital P(hit) -> position-value adjusted (data-derived premiums; superflex
     enters via QB replacement depth) -> historical percentile = score 1-100
  5. full-LR P(hit) gives the 'lean vs capital' and top-3 drivers
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


def _hit_vor() -> dict:
    """{pos: mean league-scoring VOR (ppg) of HIT seasons} over the lineup
    replacement level (config board.replacement_rank -- superflex enters via
    QB's ~24 starters). This is the value-of-a-hit in this league's points."""
    cfg = config()
    ppg = pd.read_parquet(ROOT / "data/interim/season_ppg.parquet")
    q = ppg[ppg.games >= cfg["labels"]["min_games"]]
    vor = {}
    for pos, n in cfg["board"]["replacement_rank"].items():
        d = q[q.position == pos]
        repl = d.groupby("season").ppg.apply(lambda s, k=n: s.nlargest(k).iloc[-1])
        hits = d[d.is_topN]
        vor[pos] = float((hits.ppg - hits.season.map(repl)).mean())
    return vor


def _premiums(hit_vor: dict) -> dict:
    """Positional value weights for the cross-position score: hit VOR
    normalized so the non-QB mean is 1.0. Replaces the old hand-set
    superflex_qb_premium: expected value = P(hit) x value-of-hit, and the
    data measures the value-of-hit."""
    base = np.mean([v for p, v in hit_vor.items() if p != "QB"])
    return {p: round(float(v / base), 3) for p, v in hit_vor.items()}


def _percentile_scores(feats: pd.DataFrame, frozen: dict, prems: dict) -> pd.DataFrame:
    """Capital P(hit) -> position-value-adjusted -> historical percentile (1-100).

    The historical pool is every training player's frozen train_phit, adjusted
    by the same positional premiums, so a 2026 score means what it meant
    historically.
    """
    def adj(p, pos):
        return p * prems[pos]

    hist = np.concatenate([adj(frozen[pos]["train_phit"], pos) for pos in frozen])
    hist_sorted = np.sort(hist)

    out = feats.copy()
    cap, full, lean = [], [], []
    for r in feats.itertuples():
        fp = frozen[r.position]
        row = feats.loc[[r.Index]]
        cap_p = fp["model"].predict_proba(
            row[fp["cols"]].astype(float).fillna(0).values)[:, 1][0]
        full_p = fp["full_lr"].predict_proba(
            row[fp["full_cols"]].astype(float).fillna(0).values)[:, 1][0]
        cap.append(cap_p); full.append(full_p); lean.append(full_p - cap_p)
    out["p_hit"] = cap
    out["p_hit_full"] = full
    out["lean"] = lean
    adj_val = np.array([adj(p, pos) for p, pos in zip(cap, out.position)])
    pct = np.searchsorted(hist_sorted, adj_val, side="right") / len(hist_sorted)
    out["score_1_100"] = np.clip(np.round(pct * 100).astype(int), 1, 100)
    return out


def _tier(score: int) -> int:
    for band in config()["board"]["tiers"]:
        if score >= band["min"]:
            return band["tier"]
    return config()["board"]["tiers"][-1]["tier"]


def _drivers(feats: pd.DataFrame, frozen: dict) -> list:
    """Top-3 full-LR drivers per player: standardized coef * standardized value."""
    all_drivers = []
    for r in feats.itertuples():
        fp = frozen[r.position]
        pipe = fp["full_lr"]
        scaler, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
        cols = fp["full_cols"]
        x = feats.loc[[r.Index], cols].astype(float).fillna(0.0).values[0]
        xz = (x - scaler.mean_) / scaler.scale_
        contrib = lr.coef_[0] * xz          # signed log-odds contribution
        order = np.argsort(-np.abs(contrib))[:3]
        all_drivers.append([
            {"name": DRIVER_LABELS.get(cols[i], cols[i]),
             "direction": "positive" if contrib[i] >= 0 else "negative",
             "magnitude": round(float(abs(contrib[i])), 3),
             # specificity for the drawer: the player's raw value and how far
             # from the position's training norm it sits (z, scaler units)
             "value": None if pd.isna(feats.loc[r.Index, cols[i]]) else round(float(x[i]), 2),
             "z": round(float(xz[i]), 2)}
            for i in order])
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
    frozen = joblib.load(PROC / "models.joblib")
    col = college.build()
    members = _score_members()
    # fill combine gaps with TRAINING medians (what the model learned on)
    train_feats = pd.read_parquet(ROOT / "data/interim/features.parquet")
    feats = assemble(members, col, medians=combine_medians(train_feats)).reset_index(drop=True)
    _check_combine_coverage(feats)
    feats = feats.merge(
        members[["gsis_id", "cfbd_player_id", "match_method", "college"]],
        on="gsis_id", how="left")

    hit_vor = _hit_vor()
    prems = _premiums(hit_vor)
    print(f"positional premiums (data-derived): {prems}")
    feats = _percentile_scores(feats, frozen, prems)
    # expected VOR: P(hit) x value-of-a-hit at the position, in this league's
    # ppg over replacement -- the draft-day "points" comparison across positions
    feats["exp_vor"] = [round(p * hit_vor[pos], 2)
                        for p, pos in zip(feats.p_hit, feats.position)]
    feats["tier"] = feats.score_1_100.map(_tier)
    # attach per-player drivers + college line as columns (index-aligned pre-sort)
    feats["drivers"] = _drivers(feats, frozen)
    feats["comps"] = _comps(feats, frozen)
    lines = _college_line(feats, col)
    feats["college_stats"] = [lines.get(i, {}) for i in feats.index]
    feats = _ktc(feats)
    feats = feats.merge(_draft_class()[["gsis_id", "round"]].drop_duplicates("gsis_id"),
                        on="gsis_id", how="left")

    feats = feats.sort_values("score_1_100", ascending=False, kind="stable").reset_index(drop=True)
    return pd.DataFrame({
        "player_id": feats.gsis_id, "name": feats.player, "pos": feats.position,
        "school": feats.college, "pick": feats.pick.astype(int), "round": feats["round"],
        "age": feats.age, "score_1_100": feats.score_1_100, "p_hit": feats.p_hit.round(4),
        "exp_vor": feats.exp_vor,
        "lean": feats.lean.round(4), "tier": feats.tier, "ktc_rank": feats.ktc_rank,
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
    flat = board.drop(columns=["drivers", "college_stats", "comps", "market_ranks"])
    flat.to_csv(PROC / "board.csv", index=False)
    # app reads from app/public at build time
    app_data = ROOT / "app" / "public"
    if app_data.exists():
        (app_data / "board.json").write_text(json.dumps(records))

    yr = config()["seasons"]["score_class"]
    print(f"scored {len(board)} players in the {yr} class")
    print(board.head(15)[["name", "pos", "pick", "score_1_100", "p_hit", "lean", "tier"]]
          .to_string(index=False))
    has_college = board.college_stats.map(bool).mean()
    print(f"\ncollege stat line present: {has_college:.0%} | tiers: "
          f"{board.tier.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
