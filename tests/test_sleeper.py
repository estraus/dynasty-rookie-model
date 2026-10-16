"""League-view logic — hermetic (synthetic fixtures, no Sleeper/DD calls).

Covers the pure functions in src.ingest.sleeper: consensus median,
starter-weighted strength, ranks, traded-pick ownership mapping, and the
trade-match finder.
"""
from __future__ import annotations

import pandas as pd

from src.ingest import sleeper
from src.labels.build_labels import FUMBLE_COLS, SCORING, TWOPT_COLS


def _weekly(rows):
    """Minimal nflverse-weekly-shaped frame; unspecified stat cols = 0."""
    base = {c: 0 for c in list(SCORING) + FUMBLE_COLS + TWOPT_COLS}
    return pd.DataFrame([{**base, "season": 2024, "season_type": "REG", **r}
                         for r in rows])


def _qb_weeks(pid, name, tds_per_wk, yds=250):
    return [{"player_id": pid, "player_display_name": name, "position": "QB",
             "week": w, "passing_yards": yds, "passing_tds": tds_per_wk}
            for w in range(1, 18)]


def _wr_weeks(pid, name, recs, yds):
    return [{"player_id": pid, "player_display_name": name, "position": "WR",
             "week": w, "receptions": recs, "receiving_yards": yds}
            for w in range(1, 18)]


def test_league_multipliers_qb_up_wr_down():
    # 14 QBs (replacement = 12th) and 26 WRs (replacement = 24th). The star QB
    # is TD-heavy -> gains under 6pt pass TD; the star WR is reception-heavy ->
    # loses under half-PPR vs the market's full-PPR assumption.
    rows = []
    rows += _qb_weeks("qb0", "Star QB", tds_per_wk=3, yds=300)
    for i in range(1, 14):
        rows += _qb_weeks(f"qb{i}", f"QB {i}", tds_per_wk=1, yds=250 - 5 * i)
    rows += _wr_weeks("wr0", "Star WR", recs=9, yds=95)
    for i in range(1, 26):
        rows += _wr_weeks(f"wr{i}", f"WR {i}", recs=4, yds=70 - i)
    m = sleeper.league_multipliers(sleeper.player_vor(_weekly(rows)))
    assert m["star qb"] > 1.0, m
    assert m["star wr"] < 1.0, m
    # sub-replacement players get no multiplier (stay at market value)
    assert "wr 25" not in m


def test_league_multipliers_small_sample_neutral():
    rows = []
    for i in range(14):
        rows += _qb_weeks(f"qb{i}", f"QB {i}", tds_per_wk=1)
    # star QB with only 5 games -> below MULT_MIN_GAMES -> excluded
    rows += _qb_weeks("qbx", "Tiny Sample", tds_per_wk=4)[:5]
    m = sleeper.league_multipliers(sleeper.player_vor(_weekly(rows)))
    assert "tiny sample" not in m


def test_market_scoring_convention():
    s = sleeper._market_scoring()
    assert s["passing_td"] == 4.0 and s["reception"] == 1.0
    assert s["rushing_td"] == 6.0          # everything else stays league


def _p(pos, med, name="X", age=25, vor=0.0):
    return {"name": name, "pos": pos, "age": age, "vals": {}, "med": med,
            "mult": 1.0, "adj": med, "vor": vor}


def test_win_now_sums_positive_starter_vor():
    teams = {
        "a": [_p("QB", 100, vor=8.0), _p("QB", 90, vor=3.0),
              _p("QB", 80, vor=6.0),                    # 3rd QB benched at depth 2
              _p("RB", 70, vor=0.0)],                   # replacement-level RB adds 0
        "b": [_p("QB", 100, vor=0.0), _p("RB", 90, vor=2.5), _p("RB", 80, vor=1.5)],
    }
    wn = sleeper.win_now(teams, {"QB": 2, "RB": 2})
    assert wn["a"] == 14.0        # 8 + 6 (best two QBs), RB contributes 0
    assert wn["b"] == 4.0         # 2.5 + 1.5


def _teams():
    # a: QB-rich, RB-poor. b: QB-poor, RB-rich. c: middling.
    return {
        "a": [_p("QB", 8000, "Q1"), _p("QB", 7000, "Q2"),
              _p("QB", 5000, "Q3"),               # 3rd QB ignored at depth 2
              _p("RB", 1000, "R1")],
        "b": [_p("QB", 1000, "Q4"), _p("RB", 7000, "R2"), _p("RB", 6000, "R3")],
        "c": [_p("QB", 4000, "Q5"), _p("RB", 3000, "R4")],
    }


DEPTH = {"QB": 2, "RB": 2}


def test_fc_params_derived_from_league_settings():
    league = {"roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "BN"],
              "total_rosters": 12, "scoring_settings": {"rec": 0.5}}
    p = sleeper.fc_params(league)
    assert p["numQbs"] == 2               # 1 QB slot + 1 superflex
    assert p["numTeams"] == 12
    assert p["ppr"] == 0.5


def test_fc_params_one_qb_league():
    league = {"roster_positions": ["QB", "RB", "WR", "BN"],
              "total_rosters": 10, "scoring_settings": {}}
    p = sleeper.fc_params(league)
    assert p["numQbs"] == 1               # no superflex slot
    assert p["ppr"] == 1                  # sleeper default when rec unset


def test_consensus_median_ignores_untracked_sources():
    # zeros mean "source doesn't track the player" -- excluded from the median
    assert sleeper.consensus({"a": 100, "b": 300, "c": 0}) == 200
    assert sleeper.consensus({"a": 100, "b": 200, "c": 900}) == 200  # robust to outlier
    assert sleeper.consensus({"a": 0, "b": 0}) == 0


def test_strength_sums_top_k_only():
    s = sleeper.strength(_teams(), DEPTH)
    assert s["a"]["QB"] == 15000          # 8000+7000, third QB excluded
    assert s["a"]["RB"] == 1000
    assert s["b"]["RB"] == 13000


def test_strength_alt_value_key():
    teams = {"a": [{"pos": "QB", "adj": 100, "other": 9000},
                   {"pos": "QB", "adj": 200, "other": 1000}]}
    s = sleeper.strength(teams, {"QB": 1}, key="other")
    assert s["a"]["QB"] == 9000           # ranked by the requested key, not adj


def test_ranks_one_is_strongest():
    r = sleeper.ranks(sleeper.strength(_teams(), DEPTH))
    assert r["a"]["QB"] == 1 and r["b"]["QB"] == 3
    assert r["b"]["RB"] == 1 and r["a"]["RB"] == 3


def test_trade_matches_need_and_surplus():
    r = sleeper.ranks(sleeper.strength(_teams(), DEPTH))
    # 3-team fixture: relax thresholds so "bottom" = rank 3, "top" = rank 1
    orig = sleeper.NEED_RANK, sleeper.SURPLUS_RANK
    sleeper.NEED_RANK, sleeper.SURPLUS_RANK = 3, 1
    try:
        m = sleeper.trade_matches(_teams(), r, me="a")
    finally:
        sleeper.NEED_RANK, sleeper.SURPLUS_RANK = orig
    # only b qualifies: QB rank 3 (needs QB), RB rank 1 (my weak spot)
    assert [x["owner"] for x in m] == ["b"]
    assert m[0]["pos"] == "RB"
    assert m[0]["pieces"][0]["name"] == "R2"     # best piece first


def test_my_picks_traded_pick_mapping():
    rosters = [{"roster_id": 1, "owner_id": "u1"}, {"roster_id": 2, "owner_id": "u2"}]
    users = [{"user_id": "u1", "display_name": "me"},
             {"user_id": "u2", "display_name": "them"}]
    draft = {"season": "2026", "draft_order": {"u1": 5, "u2": 9},
             "settings": {"rounds": 2}}
    # I traded away my round-1 pick; acquired their round-2 pick.
    traded = [{"season": "2026", "round": 1, "roster_id": 1, "owner_id": 2},
              {"season": "2026", "round": 2, "roster_id": 2, "owner_id": 1}]
    picks = sleeper.my_picks(rosters, traded, draft, users, me="me")
    assert picks == [{"round": 2, "slot": 5}, {"round": 2, "slot": 9}]


def test_my_picks_ignores_other_seasons():
    rosters = [{"roster_id": 1, "owner_id": "u1"}]
    users = [{"user_id": "u1", "display_name": "me"}]
    draft = {"season": "2026", "draft_order": {"u1": 1}, "settings": {"rounds": 1}}
    # a 2027 trade must not affect the 2026 draft
    traded = [{"season": "2027", "round": 1, "roster_id": 1, "owner_id": 1}]
    assert sleeper.my_picks(rosters, traded, draft, users, me="me") == [
        {"round": 1, "slot": 1}]


def test_age_weight_scales_with_age():
    # 29yo QB: value mostly current production -> near-full tilt
    assert sleeper.age_weight(29, "QB") > 0.85
    # 22yo WR: value mostly future -> floor-ish tilt
    assert sleeper.age_weight(22, "WR") == sleeper.AGE_FLOOR
    assert sleeper.age_weight(35, "RB") == 1.0        # capped at 1
    assert sleeper.age_weight(None, "QB") == 1.0      # unknown age: full


def test_age_weight_dampens_multiplier_for_youth():
    # same raw mult, old vs young: old player's adj moves more
    players = {"players": ["1", "2"]}
    db = {"1": {"full_name": "Old Guy", "position": "TE", "age": 30},
          "2": {"full_name": "Young Guy", "position": "TE", "age": 22}}
    markets = {"ktc": {"old guy": 4000, "young guy": 4000}}
    mults = {"old guy": 0.75, "young guy": 0.75}
    teams = sleeper.build_teams(
        [{"roster_id": 1, "owner_id": "u", "players": ["1", "2"]}],
        [{"user_id": "u", "display_name": "o"}], db, markets, mults)
    by = {p["name"]: p for p in teams["o"]}
    assert by["Old Guy"]["adj"] < by["Young Guy"]["adj"]
    assert by["Young Guy"]["adj"] < 4000               # still partially applied


def _pv(pos, med, adj, name):
    return {**_p(pos, med, name), "adj": adj}


def _trade_setup():
    # my TE is market-overpriced for this league (med 6800 > adj 5100 --
    # McBride-shaped); their RBs carry the usual RB haircut (adj < med).
    teams = {
        "me": [_pv("TE", 6800, 5100, "My TE"), _pv("QB", 7000, 9400, "My QB"),
               _p("RB", 1000, "My Scrub")],
        "them": [_pv("TE", 800, 800, "Their TE"), _pv("RB", 6500, 5720, "Their RB1"),
                 _pv("RB", 5800, 5100, "Their RB2")],
    }
    ranks = {"me": {"TE": 1, "QB": 1, "RB": 2}, "them": {"TE": 2, "QB": 1, "RB": 1}}
    return teams, ranks


def test_propose_trades_both_dashboards_yes():
    teams, ranks = _trade_setup()
    orig = sleeper.NEED_RANK, sleeper.SURPLUS_MAX_RANK
    sleeper.NEED_RANK, sleeper.SURPLUS_MAX_RANK = 2, 1
    try:
        props = sleeper.propose_trades(teams, ranks, "me")
    finally:
        sleeper.NEED_RANK, sleeper.SURPLUS_MAX_RANK = orig
    assert props, "expected at least one proposal"
    for pr in props:
        assert pr["my_net_adj"] > 0                    # I gain league-adj value
        assert pr["owner"] == "them"
    # the overpriced TE -> their RB1 swap must surface: they see ~fair on med
    # (6800 vs 6500) while I net adj (5720 vs 5100)
    assert any(pr["give"] == ["My TE"] and "Their RB1" in pr["get"] for pr in props)
    # my QB must NOT be shipped cheap: adj 9400 out never beats RB adj coming back
    assert not any("My QB" in pr["give"] for pr in props)


def test_propose_trades_no_needs_no_proposals():
    teams, ranks = _trade_setup()
    # I'm top-3 everywhere -> no need positions -> nothing to hunt
    flat = {"me": {"TE": 1, "QB": 1, "RB": 1}, "them": {"TE": 12, "QB": 12, "RB": 12}}
    assert sleeper.propose_trades(teams, flat, "me") == []


def test_history_snapshot_and_trend(tmp_path, monkeypatch):
    monkeypatch.setattr(sleeper, "HISTORY", tmp_path / "values.csv")
    teams = {"o": [_p("QB", 5000, "A")]}
    h1 = sleeper.snapshot_history(teams, "2026-07-01")
    assert h1.date.nunique() == 1
    # same-day rerun is idempotent
    h1b = sleeper.snapshot_history(teams, "2026-07-01")
    assert len(h1b) == 1
    # later snapshot at a higher price -> positive trend vs the old one
    teams2 = {"o": [_p("QB", 5400, "A")]}
    h2 = sleeper.snapshot_history(teams2, "2026-07-15")
    sleeper.attach_trends(teams2, h2, "2026-07-15")
    assert teams2["o"][0]["trend"] == 400
    # snapshots older than 30 days are ignored for the trend base
    teams3 = {"o": [_p("QB", 6000, "A")]}
    h3 = sleeper.snapshot_history(teams3, "2026-09-01")
    sleeper.attach_trends(teams3, h3, "2026-09-01")
    assert teams3["o"][0]["trend"] == 0
