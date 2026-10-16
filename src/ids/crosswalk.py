"""College<->NFL id crosswalk.

Match every drafted QB/RB/WR/TE (2013-2023 classes) to its CFBD college
player_id. The NFL draft table's `cfb_player_id` is a PFR slug, not a CFBD id,
so we match on normalized name within the pre-draft season window, disambiguate
ties by fuzzy school, and fall back to fuzzy name matching.

Outputs:
  data/interim/crosswalk.parquet  one row per draftee with cfbd_player_id + method
  overrides/id_matches.csv        misses, for manual fill (round-tripped on rerun)

Run:  python -m src.ids.crosswalk
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from src.util import ROOT, config

# window: a draftee's last college season is draft_year-1; look back 6 years
# to cover redshirts/transfers.
LOOKBACK = 6
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
FUZZY_NAME_MIN = 0.86      # accept a fuzzy name match at/above this
SCHOOL_TIE_MARGIN = 0.15   # school-fuzz gap needed to break a same-name tie

# school abbreviation expansions so "Florida St." ~ "Florida State"
SCHOOL_FIXES = [
    (r"\bst\b", "state"), (r"\bcol\b", "college"), (r"\buniv\b", "university"),
    (r"\bla\b", "louisiana"), (r"\bso\b", "southern"), (r"\bmiss\b", "mississippi"),
    (r"\bcent\b", "central"), (r"\bintl\b", "international"),
]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_name(s: str) -> str:
    s = _strip_accents(str(s)).lower()
    # apostrophes/periods deleted (not spaced): De'Von ~ Devon, D.K. ~ DK --
    # cross-source spelling variants collapse to one key. Both sides of every
    # match use this same function, so the change is symmetric.
    s = re.sub(r"['.’]", "", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    toks = [t for t in s.split() if t not in NAME_SUFFIXES]
    return " ".join(toks)


def norm_school(s: str) -> str:
    s = _strip_accents(str(s)).lower()
    s = re.sub(r"\([^)]*\)", " ", s)          # drop "(GA)", "(PA)"
    s = re.sub(r"[^a-z\s]", " ", s)
    for pat, rep in SCHOOL_FIXES:
        s = re.sub(pat, rep, s)
    return " ".join(s.split())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _cfbd_directory() -> pd.DataFrame:
    """One row per (cfbd player_id): name + set of seasons + normalized schools."""
    cs = pd.read_parquet(ROOT / "data/raw/cfbd_player_season.parquet")
    cs = cs[["player_id", "player", "team", "season"]].drop_duplicates()
    cs["norm_name"] = cs["player"].map(norm_name)
    cs["norm_school"] = cs["team"].map(norm_school)
    agg = cs.groupby("player_id").agg(
        player=("player", "first"),
        norm_name=("norm_name", "first"),
        seasons=("season", lambda s: set(s)),
        schools=("norm_school", lambda s: set(s)),
        teams=("team", lambda s: sorted(set(s))),
    ).reset_index()
    return agg


def _draftees() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/raw/nfl_draft.parquet")
    cfg = config()
    lo, hi = cfg["seasons"]["train_classes"]
    d = d[(d.season.between(lo, hi)) & (d.position.isin(cfg["positions"]))].copy()
    d["norm_name"] = d["pfr_player_name"].map(norm_name)
    d["norm_school"] = d["college"].map(norm_school)
    return d


def _load_overrides() -> dict:
    """Manual gsis_id -> cfbd player_id fixes. Blank ids (unfilled misses) ignored."""
    path = ROOT / config()["paths"]["overrides"]
    if not path.exists():
        return {}
    ov = pd.read_csv(path, dtype=str).fillna("")
    return {r.gsis_id: r.cfbd_player_id for r in ov.itertuples()
            if r.gsis_id and r.cfbd_player_id}


def _best_school(draft_school: str, cand) -> tuple[float, float]:
    """Best fuzzy school score for a candidate (max over its schools) and runner gap."""
    scores = sorted((_ratio(draft_school, s) for s in cand["schools"]), reverse=True)
    return (scores[0] if scores else 0.0)


def _match_one(row, by_name: dict, dir_df: pd.DataFrame):
    window = set(range(row.season - LOOKBACK, row.season))  # [yr-6 .. yr-1]
    cands = by_name.get(row.norm_name, [])
    cands = [c for c in cands if c["seasons"] & window]

    if len(cands) == 1:
        c = cands[0]
        sc = _best_school(row.norm_school, c)
        return c["player_id"], "name+year", round(sc, 3), c["player"]

    if len(cands) > 1:  # same name in window -> break tie by school
        ranked = sorted(cands, key=lambda c: _best_school(row.norm_school, c), reverse=True)
        top = _best_school(row.norm_school, ranked[0])
        second = _best_school(row.norm_school, ranked[1])
        if top - second >= SCHOOL_TIE_MARGIN:
            return ranked[0]["player_id"], "name+year+school", round(top, 3), ranked[0]["player"]
        return None  # genuinely ambiguous -> miss

    # no exact-name candidate: fuzzy name among window players at a matching school
    school_ok = dir_df[dir_df["seasons"].map(lambda s: bool(s & window))]
    best = None
    lastname_hits = []
    draft_last = row.norm_name.split()[-1] if row.norm_name else ""
    for c in school_ok.itertuples():
        sc = max((_ratio(row.norm_school, s) for s in c.schools), default=0.0)
        if sc < 0.80:
            continue
        nm = _ratio(row.norm_name, c.norm_name)
        if nm >= FUZZY_NAME_MIN and (best is None or nm > best[2]):
            best = (c.player_id, "fuzzy", round(nm, 3), c.player)
        # nickname tier: same last name + strong school; kept only if unique
        cand_last = c.norm_name.split()[-1] if c.norm_name else ""
        if sc >= 0.85 and draft_last and cand_last == draft_last:
            lastname_hits.append((c.player_id, c.player, sc))
    if best:
        return best
    if len(lastname_hits) == 1:  # unique last-name-at-school -> nickname/spelling variant
        pid, nm, sc = lastname_hits[0]
        return pid, "lastname+school", round(sc, 3), nm
    return None


def match_draftees(draftees: pd.DataFrame) -> pd.DataFrame:
    """Match a prepared draftees frame (needs norm_name, norm_school, gsis_id,
    pfr_player_name, position, college, season, round, pick) to CFBD ids."""
    dir_df = _cfbd_directory()
    by_name: dict = {}
    for c in dir_df.to_dict("records"):
        by_name.setdefault(c["norm_name"], []).append(c)

    overrides = _load_overrides()
    id_to_name = dict(zip(dir_df.player_id, dir_df.player))

    rows = []
    for row in draftees.itertuples():
        if row.gsis_id in overrides:
            pid = overrides[row.gsis_id]
            rows.append((pid, "override", 1.0, id_to_name.get(pid, "")))
            continue
        m = _match_one(row, by_name, dir_df)
        rows.append(m if m else (None, "miss", 0.0, ""))

    out = draftees[["season", "round", "pick", "gsis_id", "pfr_player_id",
                    "pfr_player_name", "position", "college"]].copy()
    out[["cfbd_player_id", "match_method", "match_score", "matched_name"]] = pd.DataFrame(
        rows, index=out.index)
    return out


def build() -> pd.DataFrame:
    return match_draftees(_draftees())


def _write_overrides_stub(out: pd.DataFrame) -> None:
    """Append unfilled misses to overrides CSV without clobbering manual fills."""
    path = ROOT / config()["paths"]["overrides"]
    existing = pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame(
        columns=["gsis_id", "pfr_player_name", "college", "season", "cfbd_player_id", "note"])
    filled = set(existing[existing.cfbd_player_id != ""].gsis_id) if len(existing) else set()
    misses = out[out.cfbd_player_id.isna()]
    new = pd.DataFrame({
        "gsis_id": misses.gsis_id, "pfr_player_name": misses.pfr_player_name,
        "college": misses.college, "season": misses.season.astype(str),
        "cfbd_player_id": "", "note": "",
    })
    new = new[~new.gsis_id.isin(filled)]
    combined = pd.concat([existing[existing.cfbd_player_id != ""], new], ignore_index=True)
    combined = combined.drop_duplicates(subset="gsis_id", keep="first")
    combined.to_csv(path, index=False)


def main() -> None:
    out = build()
    (ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
    out.to_parquet(ROOT / "data/interim/crosswalk.parquet", index=False)
    _write_overrides_stub(out)

    n = len(out)
    matched = out.cfbd_player_id.notna().sum()
    print(f"Matched {matched}/{n} ({100*matched/n:.1f}%)")
    print("by method:", out.match_method.value_counts().to_dict())
    for pos in config()["positions"]:
        p = out[out.position == pos]
        pm = p.cfbd_player_id.notna().sum()
        print(f"  {pos}: {pm}/{len(p)} ({100*pm/len(p):.1f}%)")
    misses = n - matched
    print(f"{misses} misses written to {config()['paths']['overrides']} for manual fill")


if __name__ == "__main__":
    main()
