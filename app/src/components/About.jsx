import React from "react";
import { Tape } from "./ui.jsx";

// plain-language key for every non-obvious number in the app
const SECTIONS = [
  ["Score (1–100)", [
    ["Score", "Calibrated P(hit) mapped to a historical percentile, so an 80 this year means what an 80 meant in any past class. Not a trade value — a hit-probability rank."],
    ["P(hit)", "Probability of a top-N positional season (QB12 / RB24 / WR24 / TE12 in points per game, min 8 games) within the player's first 3 NFL seasons, under this league's scoring."],
    ["Tier / cliff", "Score bands. The number on a tier rule is the score drop from the last player above the line to the first below it — how much you lose by waiting one more pick."],
    ["Slot", "NFL draft capital: round.overall. The model's strongest single input, on purpose — capital beats production models alone, so ours must beat capital to ship."],
  ]],
  ["Model vs market", [
    ["Lean", "Full production/age model minus the capital-only baseline, in P(hit). ▲ means the drivers like him beyond his draft slot; ▼ means the score is riding on capital."],
    ["KTC Δ", "KTC's rank of the class minus the model's rank. Positive = the model is higher than the market = target; negative = fade at cost."],
    ["The rank ladder", "Draft capital sets the anchor rank. The model deviates from it via drivers or positional hit rates at that slot; KTC deviates from it via market sentiment. The gap between those two deviations is the whole edge."],
    ["Other markets", "DynastyProcess, DynastySuperflex, Dynasty Daddy ADP, and DynastyDaddy ranks of the same class — context for whether KTC is the outlier or the consensus."],
  ]],
  ["Reasoning", [
    ["Top drivers", "The three largest signed contributions in the frozen logistic model. z is how far the raw stat sits from the position's training norm; 'missing — imputed' means the stat didn't exist (e.g. no combine) and the missing-flag itself is the signal."],
    ["Comps", "Nearest historical players in model feature space, with dimensions weighted by the model's own coefficients — similar where it matters, not lookalike stats trivia. HIT/MISS is the actual top-N outcome; peak is their best positional finish."],
  ]],
  ["Draft day", [
    ["xVOR", "P(hit) × the average league points-per-week over replacement that a hit returns at that position. The cross-position number: compare a QB's xVOR to a RB's directly."],
    ["Taken checkboxes", "Stored in your browser only; survive refresh. Reset draft clears them."],
  ]],
  ["League view", [
    ["Value", "Median of six market sources, repriced to this league's actual scoring (6pt pass TD, half-PPR, TE +0.25/rec) using multipliers computed from real NFL weeks scored both ways, then age-weighted."],
    ["×mult", "That repricing multiplier. >1 = this league values him more than the market convention does (QBs, TEs here); <1 = less."],
    ["Win-now", "Projected starter points per game over replacement — a lineup-strength meter, independent of dynasty value."],
    ["Trade matches / proposals", "Matches: teams strong where you're weak that need QB. Proposals: packages a market-value dashboard scores as fair for them while you gain league-adjusted value."],
  ]],
  ["Data & discipline", [
    ["Freshness", "Market values refresh daily at 11:00 UTC via CI. Rookie model inputs are frozen at draft time: college stats, combine, slot, age — no camp hype, ever."],
    ["Frozen model", "The model was frozen before scoring this class. Scores never get hand-tuned because one feels wrong."],
  ]],
];

export default function About() {
  return (
    <div className="max-w-2xl space-y-7">
      {SECTIONS.map(([title, rows]) => (
        <div key={title}>
          <div className="mb-3">
            <Tape>{title}</Tape>
          </div>
          <dl className="space-y-3">
            {rows.map(([term, def]) => (
              <div key={term} className="flex gap-4 text-sm">
                <dt className="w-32 shrink-0 font-display font-semibold tracking-wide">{term}</dt>
                <dd className="text-dim leading-relaxed">{def}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  );
}
