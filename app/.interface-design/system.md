# Dynasty Board — War Room design system

**Direction:** NFL front-office war room on draft night. Warm walnut-dark board, paper ink,
one brass accent, rows as magnetic name cards, tier breaks as tape rules that print the cliff size.

## Tokens (Tailwind v4 `@theme` in `src/index.css`)
- Surfaces: `board #161209` (page) · `card #1e1910` (rows/panels) · `raised #272013` (hover/active) · `well #100d06` (inputs — inset, darker)
- Text: `ink #eae2d0` · `dim #a99e85` · `mute #6e6552` (3-step warm paper ramp)
- Accent: `brass #d3a94f` — active nav, elite scores (≥88), selection. One accent only.
- Semantic: `up #93a86a` (olive) · `down #c9704f` (brick)
- Position magnet colors (edge tick + text tag ONLY, never filled pills): `qb #c07a54` · `rb #8fa06a` · `wr #7c94a8` · `te #c2a05a`
- Borders: `line rgba(234,226,208,.09)` · `line-strong rgba(234,226,208,.22)`

## Type
- `font-display` = Barlow Condensed 500/600/700 — player names, headers, big numerals (scores). Names get `tracking-wide`; mastheads uppercase.
- `font-mono` = IBM Plex Mono 400/500/600 — all meta numerals, labels, filters. Labels: 10–11px uppercase `tracking-widest`.
- Score ramp (`scoreColor` in lib.js): ≥88 brass · ≥74 ink · ≥58 dim · else mute. Single hue — no rainbow.

## Depth & density
- Borders only, no shadows. Cards: `border border-line bg-card rounded-sm`, hover `bg-raised`.
- `rounded-sm` everywhere (magnets are square-ish). Dense rows: `py-1.5`, list gap `space-y-1`.
- Content width `max-w-5xl mx-auto` — header inner content uses the same container.

## Components (`src/components/ui.jsx`)
- `PosTag pos` — mono 10px tracked position tag in magnet color.
- `Tape children note` — section header: tracked mono label + hairline + optional right-side mono note (used for tier breaks with `cliff −N`, and all section titles).
- Row card pattern: relative card + `absolute left-0 w-[3px]` position tick + name (display 600 base) + mono meta columns with fixed widths (score w-12 · slot w-14 · lean w-16 · ktc w-12).

## Rules
- Position color never fills a background. Tier identity comes from tape rules, not hue.
- All dynamic numbers `tabular-nums`.
- Nav = condensed uppercase tabs with 2px brass underline for active — no pill nav.
