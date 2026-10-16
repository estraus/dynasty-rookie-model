# Dynasty Rookie Model

Score every incoming rookie 1-100 for a **superflex, half-PPR, 6pt passing TD**
dynasty league. Score = calibrated P(hit) mapped to a historical percentile, so
an 80 in 2026 means what an 80 meant in 2021. Results serve a static React draft
board that reads `data/processed/board.json`.

## Ground rules (do not violate)
- **Draft capital is a feature, not a rival.** Model includes `log(overall pick)`.
  Ship criterion: beat a capital-only baseline on out-of-class validation.
- **Position-specific models.** ~250-450 samples/position. Baseline = regularized
  logistic regression; challenger = LightGBM with monotonic constraints
  (better pick -> better score). Nothing deeper.
- **Time-based validation only.** Rolling by draft class: train through year k,
  validate on k+1. No random splits (era leakage).
- **Point-in-time discipline.** Scoring a class uses only what existed at
  rookie-draft time: college stats, combine, draft slot, birthdate. No camp/preseason.

## Known traps
- 2024 and 2025 classes have incomplete 3-year windows. Exclude from training, or
  use year-1 partial labels with an explicit flag. Never silently mix.
- QB samples are tiny and hits are rare. Report uncertainty, lean on capital,
  resist overfitting to a handful of rushing-QB hits.
- **No leakage:** nothing dated after the NFL draft enters features.
- Deterministic seeds everywhere (`config.yaml: seed`). Cache raw API pulls so
  reruns are free.
- Freeze the model before looking at 2026 outputs. No re-tuning because a score
  feels wrong.
- Keep the app dumb: no model logic in JavaScript. Scoring-setting changes are a
  pipeline re-run, not an app feature.

## Layout
- `config.yaml` — league scoring, hit thresholds, season ranges. Single source of truth.
- `src/ingest/` — cfbd (college) + nflverse (NFL) pulls, cached to parquet in `data/raw/`.
- `src/ids/crosswalk.py` — college<->NFL id matching. Misses -> `overrides/id_matches.csv`.
- `src/features/{wr,rb,te,qb}.py` — per-position feature builders.
- `src/labels/build_labels.py` — PPG per player-season, hit flags.
- `src/model/{train,score}.py` — train/eval; score writes board.json + board.csv.
- `src/report/board.py` — board assembly.
- `app/` — Vite + React + Tailwind static draft board.
- `data/{raw,interim,processed}/` — parquet caches; raw is API truth, never edited.

## Env
Isolated venv at `.venv/`. Deps in `requirements.txt`. CFBD needs a free API key
in env var `CFBD_API_KEY` (get one at https://collegefootballdata.com/key).
nflverse needs no key.
