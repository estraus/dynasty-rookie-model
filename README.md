# Dynasty Rookie Model

> **Public demo.** This is a sanitized snapshot of a private working repo.
> League data in the League view is anonymized (team names replaced, league id
> removed), and the daily GitHub Actions refresh is omitted because it needs a
> real Sleeper league. To point it at your own league, set `sleeper.league_id`
> and `sleeper.username` in `config.yaml` and run `python -m src.ingest.refresh`.

Scores every incoming NFL rookie **1–100** for a **superflex, half-PPR, 6pt
passing-TD** dynasty league, and serves the results as a static React draft
board. The score is a calibrated hit probability mapped to a historical
percentile, so an 80 in 2026 means what an 80 meant in 2021.

See `CLAUDE.md` for the ground rules (capital-as-feature, time-based validation,
point-in-time discipline, no leakage).

## Headline finding

On out-of-class validation, a **capital-only model beats the full
production/age/athleticism model at every position** — the plan's explicit
"capital is enough" outcome. The shipped `score` is therefore the validated
capital P(hit); a secondary **lean** column (full-model P(hit) − capital P(hit))
and top-3 drivers surface where production/age disagree with draft slot.

Validation is walk-forward by draft class with a **label-realizability embargo**:
to score class *v* we train only on classes whose 3-season hit label had actually
closed by *v*'s draft (`c <= v − window_seasons`), so the metric never sees
future-realized outcomes. A paired **bootstrap** reports 80% CIs and, per
challenger, the fraction of resamples where it beats baseline; a challenger ships
only if that fraction ≥ 0.9. Embargoed baseline AUC (80% CI): QB .82 [.70, .94],
RB .79 [.72, .85], WR .85 [.81, .89], TE .88 [.81, .94]. Under this stricter test
the full models beat baseline in ≤0.7% of resamples (QB best 30%) — capital wins
decisively, not on a lucky split. QB's wide CI flags it as the position to trust
least; see `data/processed/metrics.json`.

## Setup
```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
echo 'CFBD_API_KEY=your_key' > .env      # free key: https://collegefootballdata.com/key
```

## Pipeline (each step caches; reruns are cheap)
```bash
export $(grep -v '^#' .env | xargs)

python -m src.ingest.nflverse      # 1. NFL draft/combine/weekly/ids/rosters -> parquet
python -m src.ingest.cfbd          #    college player/team season stats -> parquet
python -m src.ingest.verify        #    acceptance: all caches load in <10s

python -m src.ids.crosswalk        # 2. college<->NFL id crosswalk (96% match)
python -m src.labels.build_labels  # 3. league-scored PPG + hit flags
python -m src.features.build        # 4. per-position feature matrix
python -m src.model.train           # 5. baseline vs full, rolling CV, freeze
python -m src.model.score           # 6. score 2026 -> board.json + board.csv

cd app && npm install && npm run dev # 7. draft board (reads app/public/board.json)
```

## Tests
```bash
./.venv/bin/python -m pytest tests/ -q
```

## Layout
`config.yaml` scoring/thresholds · `src/ingest` pulls · `src/ids` crosswalk ·
`src/labels` hit labels · `src/features` builders · `src/model` train/score ·
`app/` board · `data/{raw,interim,processed}` caches (gitignored) ·
`overrides/id_matches.csv` manual id fills · `overrides/ktc_<year>.csv` optional KTC.

## KTC delta view
`python -m src.ingest.ktc` pulls today's superflex values from the Dynasty Daddy
public API, matches them to the score class, and writes `overrides/ktc_2026.csv`
(`name, ktc_rank, sf_value`). Re-run `python -m src.model.score` to refresh the
board's delta screen. (Fallback: hand-edit that CSV; the scorer only needs
`name, ktc_rank`.)
