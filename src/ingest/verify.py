"""Phase 1 acceptance check: all raw caches load from parquet in <10s.

Run:  python -m src.ingest.verify
Assumes caches are already built (python -m src.ingest.nflverse / .cfbd).
"""
from __future__ import annotations

import time

from src.ingest import cfbd, nflverse

TABLES = {
    "nfl_draft": nflverse.draft,
    "nfl_combine": nflverse.combine,
    "nfl_weekly": nflverse.weekly,
    "nfl_ids": nflverse.ids,
    "nfl_rosters": nflverse.rosters,
    "cfbd_player_season": cfbd.player_season,
    "cfbd_team_season": cfbd.team_season,
}


def main() -> None:
    t0 = time.perf_counter()
    total = 0
    for name, fn in TABLES.items():
        df = fn()  # cache hit -> just a parquet read
        total += len(df)
        print(f"  {name:20s} {len(df):>8,} rows")
    elapsed = time.perf_counter() - t0
    print(f"\nLoaded {total:,} rows across {len(TABLES)} tables in {elapsed:.2f}s")
    assert elapsed < 10, f"cache load took {elapsed:.2f}s, must be <10s"
    print("PASS: all caches load in <10s")


if __name__ == "__main__":
    main()
