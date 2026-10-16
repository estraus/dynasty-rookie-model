# Dynasty Rookie Model: Build Plan for Claude Code

## Objective
Score every incoming rookie 1-100 for a superflex, half-PPR, 6pt passing TD dynasty league, and serve the results in an interactive draft board web app. The score is a calibrated hit probability mapped to a historical percentile, so an 80 in 2026 means the same thing as an 80 in 2021.

## Design decisions (read before writing code)
1. **Draft capital is a feature, not a rival.** The model includes log(pick), and the shipping criterion is beating a capital-only baseline on out-of-class validation. Production and age earn their keep by explaining residuals, not by replacing capital.
2. **Position-specific models.** Usable training samples are roughly 250-450 per position across the 2013-2023 classes. That means regularized logistic regression as the baseline and LightGBM with monotonic constraints (better pick, better score) as the challenger. Nothing deeper.
3. **Label definition.** Hit = a top-12 QB, top-24 RB, top-24 WR, or top-12 TE PPG season (min 8 games) within the player's first 3 NFL seasons, scored with the league settings above. Model outputs P(hit); score = percentile of P(hit) against the historical distribution, scaled 1-100.
4. **Time-based validation only.** Rolling by draft class: train on classes through year k, validate on k+1. Random splits leak era information and inflate results.
5. **Point-in-time discipline.** Scoring the 2026 class uses only what existed at rookie-draft time: college stats, combine, NFL draft slot, birthdate. No camp reports, no preseason.
6. **App architecture: static frontend, no backend.** The model runs once per class, so there is nothing for a server to do. The pipeline writes `board.json`; a Vite + React app reads it at build time. Free hosting, no infra, and the app stays fully decoupled from the Python code. Re-run pipeline, regenerate JSON, redeploy.

## Data sources
- **College stats:** collegefootballdata.com API (free key, `cfbd` Python package). Player season passing/rushing/receiving plus team totals for market shares, 2010-2025.
- **NFL side:** `nfl_data_py` (nflverse). Draft picks, combine, weekly stats 2013-2025, rosters for birthdates.
- **ID crosswalk:** nflverse `import_ids()` covers NFL-side IDs. College-to-NFL matching is name + school + draft year with fuzzy fallback and a manual override CSV. This is the grindiest task in the project; budget a full session for it.
- **Sanity check (not training data):** KTC rookie rankings, pasted as a CSV. Do not build a scraper before the model works.

## Features
**Shared:** log(overall pick), age at rookie-season start, early-declare flag, final college season (for era normalization), power-conference flag.

**WR:** dominator rating (yards + TD share of team), breakout age (first season at 20%+ dominator), yards per team pass attempt, final-season vs best-season delta.

**RB:** scrimmage yards share, reception share (matters in half-PPR), TD share, speed score where combine data exists.

**TE:** age-adjusted receiving share. Expect capital to dominate here; that is fine.

**QB:** rush yards and rush TD per game, YPA and TD rate (era-normalized), career starts. Do not bake the superflex premium into the QB model itself; apply it at the cross-position board layer.

Normalize all market shares within season (league-wide pass rates drift). For transfer-portal players, compute both career and final-team shares.

Combine missingness is nonrandom (opt-outs skew good). Impute with position median and add a missing-indicator flag.

## Repo layout
```
dynasty-rookie-model/
  CLAUDE.md
  config.yaml          # league scoring, hit thresholds, season ranges
  src/
    ingest/cfbd.py
    ingest/nflverse.py
    ids/crosswalk.py
    features/wr.py rb.py te.py qb.py
    labels/build_labels.py
    model/train.py
    model/score.py     # writes board.json + board.csv
    report/board.py
  app/                 # Vite + React draft board, reads data/processed/board.json
  data/raw/ interim/ processed/
  overrides/id_matches.csv
  tests/
```

## Phases (one Claude Code session each)
1. **Scaffold + ingest.** Repo structure, config.yaml, CFBD and nflverse pulls cached to parquet. Acceptance: college seasons 2010-2025 and NFL draft/weekly data 2013-2025 load from cache in under 10 seconds.
2. **Crosswalk.** Match all drafted QB/RB/WR/TE from 2013-2023 classes to college stat lines. Acceptance: 95%+ match rate, misses written to overrides file for manual fill.
3. **Labels.** Compute PPG per player-season with league scoring, flag hits per the thresholds. Acceptance: spot-check 10 known hits and 10 known busts by hand.
4. **Features.** Per-position feature builders with unit tests pinned to known players (e.g., verify a specific WR's dominator by hand).
5. **Train + evaluate.** Capital-only baseline vs full model, rolling class CV, calibration plot, Brier/AUC/Spearman vs 3-year outcomes. Acceptance: full model beats baseline out of class, or the finding is "capital is enough," which is also a valid result.
6. **Score 2026 + export.** Freeze the model, score the class, write `board.json` and `board.csv` with scores, drivers, tiers, and KTC deltas.
7. **Draft board app.** Vite + React + Tailwind, static build, reads `board.json`. Features:
   - Board table: sort by score, pick, or KTC delta; filter by position and tier; name search.
   - Player detail drawer: score, P(hit), top-3 drivers with direction, college stat line, draft slot, age.
   - Delta view: model rank vs KTC rank, sorted by gap. This is the actionable screen; positive gaps are targets, negative gaps are fades at cost.
   - Draft-day mode: mark players taken (persist to localStorage), auto-surface best available overall and per position.
   - Tier breaks rendered visually so cliff edges are obvious mid-draft.
   Deploy to GitHub Pages or Vercel; also runs locally with `npm run dev`. Acceptance: full 2026 class renders, marking a player taken updates best-available instantly, state survives refresh.

## Data contract: board.json
One object per player: `player_id, name, pos, school, pick, round, age, score_1_100, p_hit, tier, drivers[3] (name, direction, magnitude), ktc_rank, delta_vs_ktc, college_stats{}`. The app treats this file as read-only truth; all interactivity (taken flags, filters) is client state.

## Known traps (put these in CLAUDE.md)
- 2024 and 2025 classes have incomplete 3-year windows. Exclude from training, or use year-1 partial labels with an explicit flag. Do not silently mix.
- QB samples are tiny and hits are rare. Report uncertainty, lean on capital, resist overfitting to a handful of rushing-QB hits.
- No leakage: nothing dated after the NFL draft enters features.
- Deterministic seeds everywhere; cache raw API pulls so reruns are free.
- Freeze the model before looking at 2026 outputs. No re-tuning because a score feels wrong.
- Keep the app dumb: no model logic in JavaScript. If scoring settings change, that is a pipeline re-run, not an app feature.
