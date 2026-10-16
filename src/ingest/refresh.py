"""CI-safe daily refresh: market values, league view, board KTC delta.

Designed to run on a fresh checkout with only light deps (pandas, pyarrow,
requests, pyyaml, nfl_data_py) -- no sklearn/joblib, no data/ caches:
  1. pull the last 2 published seasons of weekly stats (for VOR multipliers)
  2. rebuild app/public/league.json + history snapshot (src.ingest.sleeper)
  3. refresh overrides/ktc_<year>.csv and patch the committed
     app/public/board.json's ktc_rank / delta_vs_ktc in place -- scores and
     drivers are frozen model output and only change on a real re-score.

Run:  python -m src.ingest.refresh
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from src.ids.crosswalk import norm_name
from src.ingest import ktc, sleeper
from src.util import ROOT


def recent_weekly(n_seasons: int = 2) -> pd.DataFrame:
    """Last n published seasons of weekly stats, pulled directly (small enough
    for CI; the full 2013+ cache is only needed for training)."""
    import nfl_data_py as nfl
    frames = []
    year = date.today().year
    for y in range(year, year - 6, -1):
        try:
            frames.append(nfl.import_weekly_data([y]))
        except Exception:  # noqa: BLE001 - unpublished season, keep looking
            continue
        if len(frames) == n_seasons:
            break
    return pd.concat(frames, ignore_index=True)


def patch_board_ktc(board: list, ktc_ranks: dict) -> list:
    """Update ktc_rank/delta_vs_ktc on committed board rows. Board is
    score-sorted, so model rank = position in the list."""
    for i, row in enumerate(board):
        rank = ktc_ranks.get(norm_name(row["name"]))
        row["ktc_rank"] = rank
        row["delta_vs_ktc"] = (rank - (i + 1)) if rank is not None else None
    return board


def main() -> None:
    weekly = recent_weekly()
    print(f"weekly: {len(weekly):,} rows, seasons {sorted(weekly.season.unique())}")
    sleeper.main(weekly)

    ktc.build()   # rewrites overrides/ktc_<year>.csv from Dynasty Daddy
    yr = ktc.config()["seasons"]["score_class"]
    ranks = pd.read_csv(ROOT / "overrides" / f"ktc_{yr}.csv")
    ktc_ranks = {norm_name(n): int(r) for n, r in zip(ranks.name, ranks.ktc_rank)}
    board_path = ROOT / "app" / "public" / "board.json"
    board = patch_board_ktc(json.loads(board_path.read_text()), ktc_ranks)
    board_path.write_text(json.dumps(board))
    print(f"board.json: KTC delta refreshed for {sum(1 for r in board if r['ktc_rank'])} players")


if __name__ == "__main__":
    main()
