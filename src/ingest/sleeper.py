"""League view data: Sleeper rosters valued across six dynasty value markets.

Pulls the configured Sleeper league (rosters, users, traded picks, draft) and
values every rostered skill player across six dynasty markets: five superflex
boards via Dynasty Daddy (KTC, DynastyProcess, DynastySuperflex, DD ADP, DD's
own) plus FantasyCalc fetched directly with params derived from the live league
settings (QB slots, team count, reception scoring) -- the one market that tunes
to league shape. Per player we keep the per-source values, their median (the
consensus price), and the min-max spread. Strength ranks / totals / trade
matches all use the median -- robust to any single market's quirk. Writes
app/public/league.json; the app stays dumb.

No public market prices 6pt passing TDs or half-PPR, so the consensus is then
repriced for THIS league: each veteran's value-over-replacement is computed
from real NFL weeks under league scoring and under the market convention
(4pt pass TD, full PPR), and the median is scaled by the ratio (see
league_multipliers). adj is the league-priced value; med is the raw market.

Run:  python -m src.ingest.sleeper     # refresh league view
"""
from __future__ import annotations

import json
import statistics

import pandas as pd
import requests

from src.ids.crosswalk import norm_name
from src.ingest.ktc import DD_SOURCES, fetch_sf_values
from src.labels.build_labels import score_weekly
from src.util import ROOT, config

API = "https://api.sleeper.app/v1"
SKILL = ("QB", "RB", "WR", "TE")
# Dynasty Daddy market ids (see fetch_sf_values docstring). These markets are
# generic superflex boards -- they don't take league settings. FantasyCalc is
# fetched directly instead (below) because it DOES tune to league shape.
# DD_SOURCES imported from src.ingest.ktc (single home for the market enum)
SOURCE_ORDER = ["ktc", "fc", "dp", "ds", "adp", "dd"]
# trade-match thresholds: bottom-4 of 12 at QB = need; top-4 at pos = surplus
NEED_RANK, SURPLUS_RANK = 9, 4


def _get(path: str):
    r = requests.get(f"{API}/{path}", timeout=60)
    r.raise_for_status()
    return r.json()


def fc_params(league: dict) -> dict:
    """FantasyCalc query params derived from the live Sleeper league settings
    (not config): QB slots incl. superflex, team count, reception scoring."""
    slots = league["roster_positions"]
    return {"isDynasty": "true",
            "numQbs": slots.count("QB") + slots.count("SUPER_FLEX"),
            "numTeams": league["total_rosters"],
            "ppr": league["scoring_settings"].get("rec", 1)}


def fetch_fc_values(league: dict) -> dict:
    """{normalized name: value} from FantasyCalc, tuned to this league's shape.
    The one market that accepts league settings, so it's fetched directly
    rather than through Dynasty Daddy's fixed-settings mirror."""
    r = requests.get("https://api.fantasycalc.com/values/current",
                     params=fc_params(league), timeout=60)
    r.raise_for_status()
    return {norm_name(p["player"]["name"]): p["value"] for p in r.json()}


def consensus(vals: dict) -> int:
    """Median of the sources that actually price the player (zeros = untracked)."""
    priced = [v for v in vals.values() if v]
    return int(statistics.median(priced)) if priced else 0


# --- league-settings repricing -------------------------------------------
# Markets can't price 6pt pass TD or half-PPR. We can: score real NFL weeks
# under league scoring AND under the market convention (4pt pass TD, full PPR
# -- the dominant assumptions behind public dynasty values), take each player's
# PPG over replacement (VOR) under both, and tilt the market consensus by the
# ratio. Per-player, so a TD-heavy QB gains more than a checkdown QB and a
# target-hog RB takes the half-PPR haircut a volume grinder doesn't.
MULT_SEASONS = 2          # most recent completed seasons feeding the ratio
MULT_MIN_GAMES = 12       # below this the ratio is noise -> neutral 1.0
MULT_MIN_VOR = 1.0        # need >=1 ppg over replacement BOTH ways; tiny
                          # denominators near replacement make ratios explode
MULT_CLIP = (0.75, 1.35)  # ponytail: sanity clamp; widen if a scoring change ever exceeds it

# Age-weighting of the multiplier. The tilt measures CURRENT production; the
# share of a player's dynasty value that IS current production grows with age
# (a 23yo is priced mostly on future seasons the tilt knows nothing about).
# NOT an age discount -- the market already prices age; this only scales how
# much of the scoring tilt to apply. By AGE_FULL the value is ~all production.
AGE_FULL = {"QB": 30, "RB": 26, "WR": 27, "TE": 27}   # QBs age slowest
AGE_START = 21            # weight ramps linearly from here
AGE_FLOOR = 0.4           # even the youngest vet's value is ~40% near-term


def age_weight(age, pos: str) -> float:
    """Fraction of the scoring multiplier to apply at this age."""
    if age is None:
        return 1.0
    w = (age - AGE_START) / (AGE_FULL.get(pos, 27) - AGE_START)
    return min(1.0, max(AGE_FLOOR, w))


def _market_scoring() -> dict:
    return {**config()["league"], "passing_td": 4.0, "reception": 1.0,
            "te_reception_bonus": 0.0}


def _replacement(seasons: pd.DataFrame, col: str) -> dict:
    """{pos: replacement ppg} = the hit_rank-th qualified player's PPG
    (config labels.hit_rank -- the league's startable cutoff), averaged over
    the window's seasons."""
    min_g, ranks = config()["labels"]["min_games"], config()["labels"]["hit_rank"]
    out = {}
    for pos, n in ranks.items():
        qual = seasons[(seasons.position == pos) & (seasons.games_s >= min_g)]
        out[pos] = qual.groupby("season")[col].apply(
            lambda s, n=n: s.nlargest(n).iloc[-1]).mean()
    return out


def player_vor(weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-player pooled window: games + PPG-over-replacement under league
    scoring (vor_lg) and the market convention (vor_mkt)."""
    w = weekly[(weekly.season_type == "REG") & weekly.position.notna()
               & weekly.position.isin(SKILL)].copy()
    w = w[w.season.isin(sorted(w.season.unique())[-MULT_SEASONS:])]
    w["lg"], w["mkt"] = score_weekly(w), score_weekly(w, _market_scoring())

    # group by player_id, not display name -- some players' names vary across
    # seasons in nflverse (De'Von/Devon Achane), which would split their sample
    g = w.groupby(["player_id", "season"]).agg(
        player=("player_display_name", "last"),
        position=("position", lambda s: s.value_counts().index[0]),
        games_s=("week", "nunique"), lg=("lg", "sum"), mkt=("mkt", "sum"),
    ).reset_index()
    g["lg_ppg"], g["mkt_ppg"] = g.lg / g.games_s, g.mkt / g.games_s
    repl_lg, repl_mkt = _replacement(g, "lg_ppg"), _replacement(g, "mkt_ppg")

    # pool the window per player (points and games summed BEFORE any ratio, so
    # a partial injury season doesn't drag an unweighted per-season mean)
    p = g.groupby("player_id").agg(
        player=("player", "last"), position=("position", "last"),
        games=("games_s", "sum"), lg=("lg", "sum"), mkt=("mkt", "sum"))
    p["vor_lg"] = p.lg / p.games - p.position.map(repl_lg)
    p["vor_mkt"] = p.mkt / p.games - p.position.map(repl_mkt)
    return p


def league_multipliers(vor: pd.DataFrame) -> dict:
    """{normalized name: consensus-value multiplier} from league-vs-market VOR."""
    ok = ((vor.games >= MULT_MIN_GAMES)
          & (vor.vor_lg >= MULT_MIN_VOR) & (vor.vor_mkt >= MULT_MIN_VOR))
    kept = vor.loc[ok]
    mult = (kept.vor_lg / kept.vor_mkt).clip(*MULT_CLIP)
    return {norm_name(n): round(float(m), 3) for n, m in zip(kept.player, mult)}


def build_teams(rosters, users, players, markets: dict, mults: dict | None = None,
                vors: dict | None = None) -> dict:
    """{owner: [{name,pos,age,vals{src:v},med,mult,adj,vor} sorted by adj desc]}
    adj = median consensus x league-settings VOR multiplier (1.0 = no NFL
    sample yet, e.g. rookies -- their market price stands). vor = league-scoring
    ppg over lineup replacement (win-now signal; 0 = at/below replacement)."""
    names = {u["user_id"]: u["display_name"] for u in users}
    mults, vors = mults or {}, vors or {}
    teams = {}
    for r in rosters:
        rows = []
        for pid in r.get("players") or []:
            p = players.get(pid)
            if not p or p.get("position") not in SKILL:
                continue
            nm = p.get("full_name") or f'{p.get("first_name", "")} {p.get("last_name", "")}'
            key = norm_name(nm)
            vals = {src: mkt.get(key, 0) for src, mkt in markets.items()}
            med = consensus(vals)
            age = p.get("age")
            # age-weighted: young players' value is future-heavy, so the
            # current-production tilt applies only partially
            mult = 1.0 + (mults.get(key, 1.0) - 1.0) * age_weight(age, p["position"])
            rows.append({"name": nm, "pos": p["position"], "age": age,
                         "vals": vals, "med": med, "mult": round(mult, 3),
                         "adj": int(round(med * mult)),
                         "vor": round(max(vors.get(key, 0.0), 0.0), 1)})
        rows.sort(key=lambda x: -x["adj"])
        teams[names.get(r["owner_id"], str(r["owner_id"]))] = rows
    return teams


def win_now(teams: dict, depth: dict) -> dict:
    """{owner: total starting-lineup VOR (league ppg over replacement)}.
    Sums each roster's top-k positive VOR per position -- how many points/week
    the roster wins by TODAY, vs dynasty value which prices the future.
    Rookies/unproven players contribute 0 until they produce."""
    s = strength(teams, depth, key="vor")
    return {o: round(sum(v.values()), 1) for o, v in s.items()}


def strength(teams: dict, depth: dict, key: str = "adj") -> dict:
    """Starter-weighted positional strength: sum of top-k values per position."""
    return {owner: {pos: sum(sorted((p[key] for p in roster if p["pos"] == pos),
                                    reverse=True)[:k])
                    for pos, k in depth.items()}
            for owner, roster in teams.items()}


def ranks(stren: dict) -> dict:
    """{owner: {pos: 1..N rank}} (1 = strongest)."""
    out = {o: {} for o in stren}
    for pos in next(iter(stren.values())):
        for i, o in enumerate(sorted(stren, key=lambda o: -stren[o][pos])):
            out[o][pos] = i + 1
    return out


def my_picks(rosters, traded, draft, users, me: str) -> list:
    """[{round, slot}] I own in the draft's season, after traded-pick mapping."""
    names = {u["user_id"]: u["display_name"] for u in users}
    rid_owner = {r["roster_id"]: r["owner_id"] for r in rosters}
    slot = {rid: draft["draft_order"].get(uid) for rid, uid in rid_owner.items()}
    season = draft["season"]
    rounds = draft.get("settings", {}).get("rounds") or 4
    # traded_picks rows: original roster_id's (season, round) now owned by owner_id (a roster_id)
    moved = {(t["season"], t["round"], t["roster_id"]): t["owner_id"] for t in traded}
    picks = []
    for rid in rid_owner:
        for rnd in range(1, rounds + 1):
            owner_rid = moved.get((season, rnd, rid), rid)
            if names.get(rid_owner[owner_rid]) == me:
                picks.append({"round": rnd, "slot": slot[rid]})
    return sorted(picks, key=lambda p: (p["round"], p["slot"] or 99))


def trade_matches(teams, rank_of, me: str) -> list:
    """Teams weak at QB but strong where I'm weak -> their top tradeable pieces."""
    my_needs = [pos for pos in rank_of[me] if rank_of[me][pos] >= NEED_RANK]
    out = []
    for owner, roster in teams.items():
        if owner == me or rank_of[owner]["QB"] < NEED_RANK:
            continue
        for pos in my_needs:
            if pos == "QB" or rank_of[owner][pos] > SURPLUS_RANK:
                continue
            out.append({"owner": owner, "qb_rank": rank_of[owner]["QB"], "pos": pos,
                        "pos_rank": rank_of[owner][pos],
                        "pieces": [p for p in roster if p["pos"] == pos][:4]})
    return sorted(out, key=lambda m: m["qb_rank"], reverse=True)


TRADE_FAIR = (0.95, 1.20)   # what they receive / what they send, on THEIR
                            # calculator (market med): near-fair up to clear win
SURPLUS_MAX_RANK = 3        # positional surplus: I rank top-3 there
SHIP_MIN_MED = 3000         # headline pieces only; scrubs aren't trade anchors
SWEETENER_MAX = 2500        # small second pieces to balance a package


def propose_trades(teams: dict, rank_of: dict, me: str, top_n: int = 10) -> list:
    """Packages where BOTH dashboards say yes: the counterparty sees fair-or-
    winning on the market median (their calculator) while I gain league-adjusted
    value. Ship candidates are positional surplus (top-3) OR market-overpriced
    for this league (med > adj -- the market pays more than the player is worth
    under our scoring). 1-2 pieces each way, counterparty must need the anchor's
    position, max 3 proposals per counterparty."""
    mine = rank_of[me]
    needs = [p for p in mine if mine[p] >= NEED_RANK]
    if not needs:
        return []
    bigs = [p for p in teams[me]
            if p["med"] >= SHIP_MIN_MED
            and (mine[p["pos"]] <= SURPLUS_MAX_RANK or p["med"] > p["adj"])][:4]
    sweeteners = [p for p in teams[me] if p["adj"] < SWEETENER_MAX][:5]

    props = []
    for owner, roster in teams.items():
        if owner == me:
            continue
        # anchor must fill a hole THEY have
        anchors = [b for b in bigs if rank_of[owner][b["pos"]] >= NEED_RANK]
        if not anchors:
            continue
        gives = [[b] for b in anchors] + [[b, s] for b in anchors for s in sweeteners]
        targets = [p for p in roster if p["pos"] in needs][:6]
        gets = [[t] for t in targets] + [
            [a, b] for i, a in enumerate(targets) for b in targets[i + 1:]]
        best = []
        for gv in gives:
            for gt in gets:
                their_in = sum(p["med"] for p in gv)      # what they receive
                their_out = max(sum(p["med"] for p in gt), 1)
                my_net = sum(p["adj"] for p in gt) - sum(p["adj"] for p in gv)
                if TRADE_FAIR[0] <= their_in / their_out <= TRADE_FAIR[1] and my_net > 0:
                    best.append({"owner": owner,
                                 "give": [p["name"] for p in gv],
                                 "get": [p["name"] for p in gt],
                                 "my_net_adj": my_net,
                                 "their_net_med": int(their_in - their_out)})
        best.sort(key=lambda p: -p["my_net_adj"])
        # dedupe: one proposal per distinct get-package
        seen, kept = set(), []
        for b in best:
            k = tuple(sorted(b["get"]))
            if k not in seen:
                seen.add(k)
                kept.append(b)
        props.extend(kept[:3])
    return sorted(props, key=lambda p: -p["my_net_adj"])[:top_n]


def build(weekly: pd.DataFrame | None = None) -> dict:
    cfg = config()["sleeper"]
    lid, me = cfg["league_id"], cfg["username"]
    league = _get(f"league/{lid}")
    rosters = _get(f"league/{lid}/rosters")
    users = _get(f"league/{lid}/users")
    traded = _get(f"league/{lid}/traded_picks")
    draft = _get(f"league/{lid}/drafts")[0]
    players = _get("players/nfl")          # ~14MB; fine for a manual refresh
    markets = {src: fetch_sf_values(market=m) for src, m in DD_SOURCES.items()}
    markets["fc"] = fetch_fc_values(league)
    markets = {src: markets[src] for src in SOURCE_ORDER}
    if weekly is None:
        weekly = pd.read_parquet(ROOT / "data/raw/nfl_weekly.parquet")
    vor = player_vor(weekly)
    mults = league_multipliers(vor)
    vors = {norm_name(n): float(v) for n, v in zip(vor.player, vor.vor_lg)}

    teams = build_teams(rosters, users, players, markets, mults, vors)
    stren = strength(teams, cfg["starter_depth"])
    rank_of = ranks(stren)
    return {
        "league_name": league["name"], "season": draft["season"], "me": me,
        "depth": cfg["starter_depth"], "sources": SOURCE_ORDER,
        "teams": teams, "strength": stren, "ranks": rank_of,
        "totals": {o: sum(p["adj"] for p in r) for o, r in teams.items()},
        "win_now": win_now(teams, cfg["starter_depth"]),
        "picks": my_picks(rosters, traded, draft, users, me),
        "matches": trade_matches(teams, rank_of, me),
        "proposals": propose_trades(teams, rank_of, me),
    }


HISTORY = ROOT / "history" / "values.csv"


def snapshot_history(teams: dict, today: str) -> pd.DataFrame:
    """Append today's per-player values to history/values.csv (idempotent per
    day: rerunning replaces today's rows). Returns the full history."""
    rows = [{"date": today, "player": p["name"], "pos": p["pos"],
             "med": p["med"], "adj": p["adj"]}
            for roster in teams.values() for p in roster]
    new = pd.DataFrame(rows)
    if HISTORY.exists():
        old = pd.read_csv(HISTORY)
        new = pd.concat([old[old.date != today], new], ignore_index=True)
    HISTORY.parent.mkdir(exist_ok=True)
    new.to_csv(HISTORY, index=False)
    return new


def attach_trends(teams: dict, hist: pd.DataFrame, today: str) -> None:
    """Per player: med change vs the oldest snapshot in the last 30 days."""
    past = hist[hist.date < today]
    if past.empty:
        return
    cutoff = (pd.Timestamp(today) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    base = (past[past.date >= cutoff].sort_values("date")
            .groupby("player").med.first())
    for roster in teams.values():
        for p in roster:
            b = base.get(p["name"])
            p["trend"] = int(p["med"] - b) if b is not None else 0


def main(weekly: pd.DataFrame | None = None) -> None:
    from datetime import date
    data = build(weekly)
    today = date.today().isoformat()
    hist = snapshot_history(data["teams"], today)
    attach_trends(data["teams"], hist, today)
    out = ROOT / "app" / "public" / "league.json"
    out.write_text(json.dumps(data))
    mine = data["ranks"][data["me"]]
    print(f'{data["league_name"]} ({data["season"]}) -> {out}')
    picks = ", ".join(f'{p["round"]}.{p["slot"]:02d}' for p in data["picks"])
    print(f"my ranks (6-source median): {mine} | picks: {picks}")
    print(f'{len(data["matches"])} trade matches, {len(data["proposals"])} proposals, '
          f'history {hist.date.nunique()} snapshot(s)')


if __name__ == "__main__":
    main()
