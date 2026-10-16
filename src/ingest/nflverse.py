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


# nflverse moved weekly player stats to a new release ("stats_player") starting
# with 2025; nfl_data_py's import_weekly_data can't read it and silently skips
# those seasons. Same stats, four renamed columns.
STATS_PLAYER_FROM = 2025
STATS_PLAYER_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                    "stats_player/stats_player_week_{year}.parquet")
STATS_PLAYER_RENAME = {"team": "recent_team", "passing_interceptions": "interceptions",
                       "sacks_suffered": "sacks", "sack_yards_lost": "sack_yards"}


def weekly_season(year: int):
    """One season of weekly player stats in the legacy column layout."""
    if year < STATS_PLAYER_FROM:
        return nfl.import_weekly_data([year])
    import io

    import pandas as pd
    import requests
    r = requests.get(STATS_PLAYER_URL.format(year=year), timeout=120)
    r.raise_for_status()
    df = pd.read_parquet(io.BytesIO(r.content)).rename(columns=STATS_PLAYER_RENAME)
    return df


def weekly_seasons(years):
    """Concat weekly stats across seasons, skipping unpublished ones (404)."""
    import pandas as pd

    frames = []
    for y in years:
        try:
            frames.append(weekly_season(y))
        except Exception as e:  # noqa: BLE001 - network/404, keep going
            print(f"  skip {y}: {str(e)[:60]}")
    return pd.concat(frames, ignore_index=True)


def weekly():
    """Completed-season weekly stats (labels). Refreshes a cache that stops
    short of nfl_end (built before the stats_player move)."""
    import pandas as pd

    end = config()["seasons"]["nfl_end"]
    df = cached("nfl_weekly", lambda: weekly_seasons(_years()))
    missing = [y for y in _years() if y > int(df.season.max())]
    if missing:  # append only the new seasons; cached ones stay byte-identical
        new = weekly_seasons(missing)
        df = cached("nfl_weekly", lambda: pd.concat(
            [df, new[[c for c in df.columns if c in new.columns]]], ignore_index=True),
            refresh=True)
    if df.season.max() < end:
        print(f"  warning: weekly stats end at {df.season.max()}, config nfl_end={end}")
    return df


def current_weekly():
    """The in-progress season (score_class), never cached: it changes weekly.
    Used only by the in-season update, never by labels."""
    return weekly_season(config()["seasons"]["score_class"])


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
