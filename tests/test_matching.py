"""Crosswalk _match_one tiers — hermetic, synthetic candidates (no data caches).

Exercises each matching path directly so a regression in the tie-break / fuzzy /
nickname logic fails here (fast, fresh-clone) rather than only as a drop in the
built crosswalk's aggregate match rate.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.ids import crosswalk as x


def _cand(pid, player, nn, seasons, schools):
    return {"player_id": pid, "player": player, "norm_name": nn,
            "seasons": set(seasons), "schools": set(schools)}


def _setup(cands):
    by_name: dict = {}
    for c in cands:
        by_name.setdefault(c["norm_name"], []).append(c)
    return by_name, pd.DataFrame(cands)


def _row(nn, school, season):
    return SimpleNamespace(norm_name=nn, norm_school=school, season=season)


def test_single_name_in_window():
    by, dd = _setup([_cand("1", "John Smith", "john smith", [2018], ["alabama"])])
    pid, method, _, _ = x._match_one(_row("john smith", "alabama", 2020), by, dd)
    assert (pid, method) == ("1", "name+year")


def test_out_of_window_is_not_a_name_match():
    # season 2030 -> window {2024..2029}; the only candidate played 2018 -> no name match.
    by, dd = _setup([_cand("1", "John Smith", "john smith", [2018], ["alabama"])])
    assert x._match_one(_row("john smith", "alabama", 2030), by, dd) is None


def test_same_name_tie_broken_by_school():
    cands = [_cand("1", "John Smith", "john smith", [2019], ["alabama"]),
             _cand("2", "John Smith", "john smith", [2019], ["ohio state"])]
    by, dd = _setup(cands)
    pid, method, _, _ = x._match_one(_row("john smith", "alabama crimson tide", 2020), by, dd)
    assert (pid, method) == ("1", "name+year+school")


def test_same_name_same_school_is_ambiguous_miss():
    # two identical-school candidates -> no school gap to break the tie -> miss,
    # never a coin-flip false positive.
    cands = [_cand("1", "John Smith", "john smith", [2019], ["alabama"]),
             _cand("2", "John Smith", "john smith", [2019], ["alabama"])]
    by, dd = _setup(cands)
    assert x._match_one(_row("john smith", "unrelated place", 2020), by, dd) is None


def test_fuzzy_name_at_matching_school():
    # spelling variant, no exact-name candidate, strong school -> fuzzy tier.
    by, dd = _setup([_cand("9", "Jonathan Smithe", "jonathon smithe", [2019], ["clemson tigers"])])
    pid, method, score, _ = x._match_one(_row("jonathon smith", "clemson tigers", 2020), by, dd)
    assert (pid, method) == ("9", "fuzzy")
    assert score >= x.FUZZY_NAME_MIN


def test_nickname_recovered_by_lastname_and_school():
    # "Tank Dell" (draft) vs "Nathaniel Dell" (college): no name/fuzzy hit, but a
    # unique last-name-at-school -> lastname+school tier.
    by, dd = _setup([_cand("5", "Nathaniel Dell", "nathaniel dell", [2022], ["houston cougars"])])
    pid, method, _, _ = x._match_one(_row("tank dell", "houston cougars", 2023), by, dd)
    assert (pid, method) == ("5", "lastname+school")
