"""College-side pulls from collegefootballdata.com REST API.

Uses the stable REST endpoints directly (Bearer key in CFBD_API_KEY) instead of
the cfbd SDK, which churns across major versions. Caches long-form stat lines to
data/raw/. Run to build:  python -m src.ingest.cfbd

Cached tables:
  cfbd_player_season  long: (season, player_id, player, team, conference,
                             category, stat_type, stat) for passing/rushing/receiving
  cfbd_team_season    long: (season, team, conference, stat_name, stat_value)
                             -- team totals, the denominators for market shares
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

from src.util import cached, config

BASE = "https://api.collegefootballdata.com"
PLAYER_CATEGORIES = ["passing", "rushing", "receiving"]


def _key() -> str:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY not set. Get a free key at "
            "https://collegefootballdata.com/key and export CFBD_API_KEY=..."
        )
    return key


def _get(path: str, params: dict, *, retries: int = 3) -> list:
    headers = {"Authorization": f"Bearer {_key()}"}
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}{path}", params=params, headers=headers, timeout=120)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {path} {params} ({str(e)[:40]})")
            time.sleep(2 * (attempt + 1))


def _college_years():
    s = config()["seasons"]
    return list(range(s["college_start"], s["college_end"] + 1))


def _pull_player_season() -> pd.DataFrame:
    rows = []
    for year in _college_years():
        for cat in PLAYER_CATEGORIES:
            data = _get("/stats/player/season", {"year": year, "category": cat})
            rows.extend(data)
            time.sleep(0.3)  # ponytail: be polite to the free API; one pull, then cached
    df = pd.DataFrame(rows)
    return df.rename(columns={"playerId": "player_id", "statType": "stat_type"})


def _pull_team_season() -> pd.DataFrame:
    rows = []
    for year in _college_years():
        data = _get("/stats/season", {"year": year})
        rows.extend(data)
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    return df.rename(columns={"statName": "stat_name", "statValue": "stat_value"})


def player_season():
    return cached("cfbd_player_season", _pull_player_season)


def team_season():
    return cached("cfbd_team_season", _pull_team_season)


def build_all():
    for name, fn in [("player_season", player_season), ("team_season", team_season)]:
        df = fn()
        print(f"  cfbd_{name:13s} {len(df):>8,} rows x {len(df.columns)} cols")


if __name__ == "__main__":
    print("Building cfbd caches (needs CFBD_API_KEY)...")
    build_all()
