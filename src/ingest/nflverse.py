"""NFL-side pulls via nfl_data_py (nflverse). No API key needed.

Caches to data/raw/: draft picks, combine, weekly stats, id crosswalk, rosters.
Run as a module to (re)build the caches:  python -m src.ingest.nflverse
"""
from __future__ import annotations

import nfl_data_py as nfl

from src.util import cached, config


def _years():
    s = config()["seasons"]
    return list(range(s["nfl_start"], s["nfl_end"] + 1))


def _by_year(importer):
    """Concat a per-year importer, skipping seasons not yet published (404).

    nflverse publishes weekly/roster files per season; a future season simply
    isn't there yet. Skip it loudly rather than hard-failing the whole pull.
    """
    import pandas as pd

    frames = []
    for y in _years():
        try:
            frames.append(importer([y]))
        except Exception as e:  # noqa: BLE001 - network/404, keep going
            print(f"  skip {y}: {str(e)[:50]}")
    return pd.concat(frames, ignore_index=True)


def draft():
    return cached("nfl_draft", lambda: nfl.import_draft_picks(_years()))


def combine():
    """Combine results through the score class, not just nfl_end.

    nfl_end bounds NFL *season* data (weekly/rosters), but the combine for the
    class being scored happens before its first season. Pulling only through
    nfl_end left every score-class player with no speed score. A cache built
    before this fix lacks the score class, so it is refreshed automatically.
    """
    s = config()["seasons"]
    years = list(range(s["nfl_start"], max(s["nfl_end"], s["score_class"]) + 1))
    df = cached("nfl_combine", lambda: nfl.import_combine_data(years))
    if df.season.max() < s["score_class"]:
        df = cached("nfl_combine", lambda: nfl.import_combine_data(years), refresh=True)
    return df


def weekly():
    # Weekly player stats -> used to compute per-season PPG in the labels step.
    return cached("nfl_weekly", lambda: _by_year(nfl.import_weekly_data))


def ids():
    # Cross-source id map (gsis, pfr, pff, sleeper, ...). No year arg.
    return cached("nfl_ids", nfl.import_ids)


def rosters():
    return cached("nfl_rosters", lambda: _by_year(nfl.import_seasonal_rosters))


def build_all():
    for name, fn in [
        ("draft", draft), ("combine", combine), ("weekly", weekly),
        ("ids", ids), ("rosters", rosters),
    ]:
        df = fn()
        print(f"  nfl_{name:8s} {len(df):>7,} rows x {len(df.columns)} cols")


if __name__ == "__main__":
    print("Building nflverse caches...")
    build_all()
