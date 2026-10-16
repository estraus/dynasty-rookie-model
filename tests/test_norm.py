"""Name/school normalization — hermetic (no data caches needed).

norm_name is the linchpin of BOTH the college<->NFL crosswalk and the KTC
name match, so a regression here silently drops matches on both. Pin the
transforms: accent strip, punctuation->space, suffix drop, school abbrev expand.
"""
from __future__ import annotations

from src.ids.crosswalk import norm_name, norm_school

NAME_CASES = {
    "Marvin Harrison Jr.": "marvin harrison",   # Jr suffix dropped
    "Michael Pittman II": "michael pittman",     # II suffix dropped
    "Robert Griffin III": "robert griffin",      # III suffix dropped
    "Ja'Marr Chase": "jamarr chase",             # apostrophe deleted
    "Amon-Ra St. Brown": "amon ra st brown",     # hyphen -> space, period deleted
    "D.K. Metcalf": "dk metcalf",                # periods deleted: D.K. ~ DK
    "José Ramírez": "jose ramirez",              # accents stripped
    "  DeAndre   Hopkins  ": "deandre hopkins",  # collapse whitespace
}

SCHOOL_CASES = {
    "Florida St.": "florida state",     # St -> state
    "Miami (FL)": "miami",              # drop parenthetical
    "Ole Miss": "ole mississippi",      # Miss -> mississippi
    "So. California": "southern california",  # So -> southern
    "LSU": "lsu",                       # untouched
}


def test_norm_name_cases():
    for raw, want in NAME_CASES.items():
        assert norm_name(raw) == want, f"{raw!r} -> {norm_name(raw)!r}, want {want!r}"


def test_norm_school_cases():
    for raw, want in SCHOOL_CASES.items():
        assert norm_school(raw) == want, f"{raw!r} -> {norm_school(raw)!r}, want {want!r}"


def test_suffix_only_names_match_across_sources():
    # the practical reason norm exists: KTC "Marvin Harrison Jr" must key-match
    # the draft table's "Marvin Harrison" (no suffix).
    assert norm_name("Marvin Harrison Jr") == norm_name("Marvin Harrison")
    assert norm_name("Kenneth Walker III") == norm_name("Kenneth Walker")
    # spelling variants across data sources collapse to one key
    assert norm_name("De'Von Achane") == norm_name("Devon Achane")
    assert norm_name("D.K. Metcalf") == norm_name("DK Metcalf")


def test_norm_name_is_idempotent():
    for raw in NAME_CASES:
        once = norm_name(raw)
        assert norm_name(once) == once
