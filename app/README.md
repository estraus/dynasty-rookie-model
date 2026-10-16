# Dynasty Draft Board (app)

Static Vite + React + Tailwind board. Reads `public/board.json` (written by the
Python pipeline, `python -m src.model.score`). No backend — re-run the pipeline,
regenerate the JSON, redeploy.

## Local
```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # -> dist/
npm run preview    # serve the production build
```

## Refresh the data
From the repo root, after re-scoring:
```bash
python -m src.model.score          # writes data/processed/board.json + app/public/board.json
```

## Views
- **Board** — sortable (score / pick / lean / KTC Δ), filter by position + tier,
  name search, visual tier breaks.
- **KTC Delta** — model rank vs KTC rank sorted by gap (needs `overrides/ktc_2026.csv`).
- **Draft Day** — check players taken (persists to localStorage), best-available
  overall + per position updates live.

## Deploy
`vite.config.js` uses `base: "./"` so relative asset paths work anywhere.
Live on **Vercel** (`vercel.json` sets framework/build/output). Redeploy after
re-scoring:
```bash
python -m src.model.score          # rewrites public/board.json
npx vercel --prod --yes            # ~5s redeploy from app/
```
