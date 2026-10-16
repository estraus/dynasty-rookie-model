"""In-season update: re-value recent rookies with their NFL production to date.

The draft-time score only uses information from April. Once a rookie plays,
his usage and points are far better evidence. This module answers, at any
point in a season: "given his draft slot AND what he has done in the NFL so
far, what is his expected 3-season value now?"

Checkpoint = (seasons completed in his window, weeks into the current one).
In October 2026 week 6: the 2026 class is at (0, 6), 2025 at (1, 6), 2024 at
(2, 6). For each position x checkpoint we fit a ridge regression of the same
`value` target on:
  prior        draft-time capital value (the value model's capital curve)
  vor_pg       league points per ELAPSED week minus replacement ppg, to date
               (missed weeks count as 0, so no-shows read as -replacement)
  tgt_share    mean weekly target share
  carry_share  mean weekly share of his team's carries
Training replays the identical checkpoint for every complete 2013+ class, so
nothing after the cutoff leaks in. The current season's replacement level isn't
known yet, so live scoring uses the prior season's (training uses each
season's own; replacement levels move little year to year).

Exported as plain coefficients (data/processed/inseason_model.json, copied to
app/public/) so the daily CI refresh can apply it with pandas only.

Run:  python -m src.model.inseason        (train + export; needs value models)
      python -m src.model.inseason apply  (score current rookies -> inseason.json)
"""
from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from src.util import ROOT, config

PROC = ROOT / "data/processed"
MODEL_PATH = PROC / "inseason_model.json"
OUT_PATH = ROOT / "app/public/inseason.json"
FEATS = ["prior", "vor_pg", "tgt_share", "carry_share"]
ALPHA = 5.0
MIN_GAIN = 0.02  # NFL-usage model must beat draft slot by this much (Spearman, OOF)
MAX_WEEK = 18


# ---------- shared by training and live apply (pandas/numpy only) ----------

def usage_table(weekly: pd.DataFrame, skill_only: bool = True) -> pd.DataFrame:
    """Per player-week: league points, target share, team carry share.

    skill_only=False keeps every row: players are matched by id, and nflverse
    sometimes lists a two-way player under his defensive position (Travis
    Hunter's 2025 WR weeks are tagged CB).
    """
    from src.labels.build_labels import score_weekly
    w = weekly[weekly.season_type == "REG"].copy()
    if skill_only:
        w = w[w.position.isin(config()["positions"])]
    w["pts"] = score_weekly(w)
    team = weekly[weekly.season_type == "REG"].groupby(["season", "week", "recent_team"]).carries.sum()
    w["team_carries"] = [team.get((s, k, t), np.nan) for s, k, t in zip(w.season, w.week, w.recent_team)]
    w["carry_share"] = (w.carries.fillna(0) / w.team_carries).where(w.team_carries > 0, 0.0)
    w["tgt_share"] = w.target_share.fillna(0.0)
    return w[["player_id", "season", "week", "position", "pts", "tgt_share", "carry_share"]]


def replacement_ppg(weekly: pd.DataFrame) -> dict:
    """{(season, pos): replacement ppg} from full-season qualified players."""
    cfg = config()
    u = usage_table(weekly)
    g = u.groupby(["player_id", "season"]).agg(pos=("position", "first"), games=("week", "nunique"),
                                              pts=("pts", "sum")).reset_index()
    g = g[g.games >= cfg["labels"]["min_games"]]
    g["ppg"] = g.pts / g.games
    out = {}
    for (season, pos), d in g.groupby(["season", "pos"]):
        n = cfg["board"]["replacement_rank"].get(pos)
        if n and len(d) >= n:
            out[(int(season), pos)] = float(d.ppg.nlargest(n).iloc[-1])
    return out


def features_at(players: pd.DataFrame, usage: pd.DataFrame, repl: dict,
                seasons_done: int, week: int, repl_lag: int = 0) -> pd.DataFrame:
    """Usage features for each player at checkpoint (seasons_done, week).

    players: gsis_id, position, draft_season, prior. Uses rookie seasons
    draft_season .. draft_season + seasons_done (that last season only through
    `week`). repl_lag=1 uses the previous season's replacement level (live).
    """
    out = players.reset_index(drop=True).copy()
    m = usage.drop(columns="position").merge(
        out[["gsis_id", "draft_season", "position"]], left_on="player_id", right_on="gsis_id")
    rel = m.season - m.draft_season
    m = m[(rel >= 0) & ((rel < seasons_done) | ((rel == seasons_done) & (m.week <= week)))]
    med = {p: float(np.nanmedian([v for (_, q), v in repl.items() if q == p]))
           for p in config()["positions"]}
    if len(m):
        m = m.assign(repl=[repl.get((int(s) - repl_lag, p), med[p])
                           for s, p in zip(m.season, m.position)])
    else:
        m = m.assign(repl=pd.Series(dtype=float))
    g = m.groupby("gsis_id").agg(pts=("pts", "sum"), games=("week", "count"), repl=("repl", "mean"),
                                 tgt_share=("tgt_share", "mean"), carry_share=("carry_share", "mean"))
    elapsed = seasons_done * 17 + min(week, 17)
    out = out.merge(g, on="gsis_id", how="left")
    out[["pts", "games", "tgt_share", "carry_share"]] = out[
        ["pts", "games", "tgt_share", "carry_share"]].fillna(0.0)
    out["repl"] = out.repl.fillna(out.position.map(med))
    if elapsed:
        # per ELAPSED week, not per game played: a missed week counts as 0
        # points, so a player who never sees the field reads as -replacement
        # and one big game in six weeks reads as one big game in six weeks
        out["vor_pg"] = out.pts / elapsed - out.repl
        out["avail"] = (out.games / elapsed).clip(upper=1.0)
    else:
        out["vor_pg"], out["avail"] = 0.0, 0.0
    return out.drop(columns=["pts", "repl"])


def predict(model: dict, X: pd.DataFrame) -> np.ndarray:
    x = (X[FEATS].astype(float).values - np.array(model["mean"])) / np.array(model["scale"])
    return np.clip(model["intercept"] + x @ np.array(model["coef"]), 0, None)


def percentile_score(dyn: np.ndarray, hist: np.ndarray) -> np.ndarray:
    pct = np.searchsorted(hist, dyn, side="right") / len(hist)
    return np.clip(np.round(pct * 100).astype(int), 1, 100)


# ------------------------------- training -----------------------------------

def _training_players() -> pd.DataFrame:
    import joblib
    vf = joblib.load(PROC / "value_models.joblib")
    lab = pd.read_parquet(ROOT / "data/interim/labels.parquet")
    lab = lab[lab.label_complete.astype(bool) & lab.gsis_id.notna()].copy()
    lab["prior"] = 0.0
    for pos, d in lab.groupby("position"):
        lab.loc[d.index, "prior"] = vf[pos]["models"]["capital"].capital(d)
    return lab, vf


def _fit_ridge(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import Ridge
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    r = Ridge(alpha=ALPHA).fit((X - mean) / scale, y)
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "intercept": float(r.intercept_), "coef": r.coef_.tolist()}


def _prior_only() -> dict:
    """Identity on the prior: value = draft-time capital value."""
    k = len(FEATS)
    return {"mean": [0.0] * k, "scale": [1.0] * k, "intercept": 0.0,
            "coef": [1.0] + [0.0] * (k - 1)}


def train() -> dict:
    from scipy.stats import spearmanr
    from src.ingest.nflverse import weekly
    cfg = config()
    window = cfg["labels"]["window_seasons"]
    wk = weekly()
    usage, repl = usage_table(wk, skill_only=False), replacement_ppg(wk)
    lab, vf = _training_players()

    from src.model.value import StackedValue
    models, report = {}, {}
    for pos in cfg["positions"]:
        P = lab[lab.position == pos].reset_index(drop=True)
        # walk-forward capital prior: what the draft-time model said, out of class
        prior_oof = pd.Series(np.nan, index=P.index)
        folds = []
        for v in sorted(P.draft_season.unique()):
            tr, va = P[P.draft_season <= v - window], P[P.draft_season == v]
            if len(tr) >= 30 and len(va):
                prior_oof[va.index] = StackedValue(None).fit(tr, tr.value.values).capital(va)
                folds.append(v)
        ok = prior_oof.notna()
        base = float(spearmanr(prior_oof[ok], P.value[ok]).correlation)
        models[pos] = {}
        for s in range(window):
            for w in range(0, MAX_WEEK + 1):
                key = f"{s}:{w}"
                F = features_at(P, usage, repl, s, w)
                oof = pd.Series(np.nan, index=F.index)
                for v in folds:
                    tr, va = F[F.draft_season <= v - window].copy(), F[F.draft_season == v].copy()
                    va["prior"] = prior_oof[va.index]  # no in-sample prior at validation
                    oof[va.index] = predict(_fit_ridge(tr[FEATS].values, tr.value.values), va)
                rho = float(spearmanr(oof[ok], F.value[ok]).correlation)
                # NFL usage ships at a checkpoint only if it beat draft slot out of class
                use = rho >= base + MIN_GAIN
                m = _fit_ridge(F[FEATS].values, F.value.values) if use else _prior_only()
                m["validation"] = {"draft_only": round(base, 3), "with_nfl": round(rho, 3), "used": use}
                models[pos][key] = m
                if w in (6, 17):
                    report[f"{pos} yr{s + 1} wk{w}"] = {"n": int(ok.sum()), **m["validation"]}

    # capital prior per pick (so live apply needs no scipy) + score pool
    picks = pd.DataFrame({"pick": np.arange(1, 263)})
    export = {
        "trained": str(date.today()), "window": window, "feats": FEATS, "models": models,
        "prior_by_pick": {pos: np.round(vf[pos]["models"]["capital"].capital(picks), 4).tolist()
                          for pos in vf},
        "horizon": {pos: vf[pos]["horizon"] for pos in vf},
        "hist": np.round(np.sort(np.concatenate([vf[p]["hist_value"] for p in vf])), 4).tolist(),
        "validation": report,
    }
    MODEL_PATH.write_text(json.dumps(export))
    (ROOT / "app/public/inseason_model.json").write_text(json.dumps(export))
    print(f"{'checkpoint':16} {'n':>4} {'draft only':>10} {'with NFL':>9}  used")
    for k, v in report.items():
        print(f"{k:16} {v['n']:>4} {v['draft_only']:>10} {v['with_nfl']:>9}  {v['used']}")
    return report


# ------------------------------ live apply -----------------------------------

def apply(current: pd.DataFrame | None = None, completed: pd.DataFrame | None = None,
          model_path=None) -> list:
    """Score the classes still inside their window, at the current week.

    current:   weekly stats for the in-progress season (score_class)
    completed: weekly stats for the completed seasons the classes need
    """
    cfg = config()
    path = model_path or (ROOT / "app/public/inseason_model.json")
    M = json.loads(path.read_text())
    window, season = M["window"], cfg["seasons"]["score_class"]
    from src.ingest import nflverse
    if current is None:
        current = nflverse.current_weekly()
    if completed is None:
        completed = nflverse.weekly_seasons(range(season - window + 1, season))
    wk = pd.concat([completed, current], ignore_index=True)
    week = int(current[current.season_type == "REG"].week.max()) if len(current) else 0

    draft = _drafts(range(season - window + 1, season + 1))
    draft["prior"] = [M["prior_by_pick"][p][min(int(k), 262) - 1] for p, k in zip(draft.position, draft.pick)]
    usage, repl = usage_table(wk, skill_only=False), replacement_ppg(completed)
    hist = np.array(M["hist"])
    rows = []
    for (cls, pos), P in draft.groupby(["draft_season", "position"]):
        s = season - int(cls)
        F = features_at(P, usage, repl, s, week, repl_lag=1).reset_index(drop=True)
        now = predict(M["models"][pos][f"{s}:{min(week, MAX_WEEK)}"], F)
        h = M["horizon"][pos]
        F["now_value"], F["now_score"] = now, percentile_score(now * h, hist)
        F["draft_score"] = percentile_score(F.prior.values * h, hist)
        u = usage[usage.player_id.isin(F.gsis_id) & (usage.season >= cls)]
        ppg = u.groupby("player_id").pts.mean()
        for r in F.itertuples():
            rows.append({
                "player_id": r.gsis_id, "name": r.player, "pos": pos, "class": int(cls),
                "pick": int(r.pick), "checkpoint": f"Year {s + 1}, week {week}",
                "draft_score": int(r.draft_score), "now_score": int(r.now_score),
                "change": int(r.now_score) - int(r.draft_score),
                "now_value": round(float(r.now_value), 2),
                "games": int(r.games),
                "ppg": round(float(ppg.get(r.gsis_id, 0.0)), 1),
                "tgt_share": round(float(r.tgt_share), 3),
                "carry_share": round(float(r.carry_share), 3),
            })
    rows.sort(key=lambda r: -r["now_score"])
    meta = {"season": season, "week": week, "updated": str(date.today()), "players": rows}
    OUT_PATH.write_text(json.dumps(meta))
    print(f"in-season: {len(rows)} players, {season} week {week} -> {OUT_PATH}")
    return rows


def _drafts(classes) -> pd.DataFrame:
    """Skill-position draft picks for the given classes (CI-safe pulls)."""
    import nfl_data_py as nfl
    d = nfl.import_draft_picks(list(classes))
    d = d[d.position.isin(config()["positions"]) & d.gsis_id.notna()]
    return d.rename(columns={"season": "draft_season", "pfr_player_name": "player"})[
        ["gsis_id", "player", "position", "draft_season", "pick"]].reset_index(drop=True)


if __name__ == "__main__":
    from src.model import inseason as _m  # package module, not __main__
    _m.apply() if sys.argv[1:] == ["apply"] else _m.train()
