import React from "react";
import { Tape } from "./ui.jsx";

// plain-language key for every non-obvious number in the app
const SECTIONS = [
  ["Score (1–100)", [
    ["Score", "Expected dynasty value mapped to a historical percentile, so an 80 this year means what an 80 meant in any past class. Value = xVOR × a position career-length multiplier (below)."],
    ["xVOR", "Expected league points per week over replacement, averaged over a player's best 2 of his first 3 NFL seasons. Replacement = the last lineup starter league-wide (QB24 for superflex, RB30, WR42, TE12). Missing or below-replacement seasons count as zero, so one hot 8-game stretch is worth half of two."],
    ["Range", "80% of past players with a similar expected value landed in this xVOR range. Wide on purpose: rookies are uncertain."],
    ["Career multiplier", "RB value is front-loaded and WR/TE value keeps coming after year 3. Measured from 2013–2020 classes: years 4–6 add 0.8× the first three seasons' value for RBs, 1.0× for WRs, 0.9× for QBs and 1.8× for TEs. Without this, a 3-season window overrates RBs."],
    ["Confidence", "How well draft slot has ranked outcomes at the position out of class (Spearman rank correlation): WR/RB ~0.5, QB ~0.45, TE ~0.38. TE scores are the least reliable."],
    ["P(hit)", "Probability of a top-N positional season (QB12 / RB24 / WR24 / TE12 in points per game, min 8 games) within the first 3 seasons. Kept for reference; the score no longer uses it."],
    ["Tier / cliff", "Score bands. The number on a tier rule is the score drop from the last player above the line to the first below it — how much you lose by waiting one more pick."],
    ["Slot", "NFL draft capital: round.overall. The model's single input, on purpose — in backtest no production or age model beat draft slot alone at any position, so the score is draft slot valued in this league's scoring."],
  ]],
  ["Model vs market", [
    ["Lean", "What a production/age model adds to or takes off the draft-slot value, in pts/wk. Faint with a ? = unvalidated: it didn't beat draft slot in backtest at that position (currently every position). A tiebreaker, not a signal."],
    ["KTC Δ", "KTC's rank of the class minus the model's rank. Positive = the model is higher than the market = target; negative = fade at cost."],
    ["The rank ladder", "Draft capital sets the anchor rank. The model deviates from it via drivers or positional hit rates at that slot; KTC deviates from it via market sentiment. The gap between those two deviations is the whole edge."],
    ["Other markets", "DynastyProcess, DynastySuperflex, Dynasty Daddy ADP, and DynastyDaddy ranks of the same class — context for whether KTC is the outlier or the consensus."],
  ]],
  ["Reasoning", [
    ["Top drivers", "The three largest contributions in pts/wk: draft capital (his slot's value vs the typical drafted player at the position), then the production/age model's per-stat adjustments. z is how far the raw stat sits from the position's training norm; 'missing — imputed' means the stat didn't exist (e.g. no combine) and the missing-flag itself is the signal."],
    ["Comps", "Nearest historical players in model feature space, with dimensions weighted by the model's own coefficients — similar where it matters, not lookalike stats trivia. HIT/MISS is the actual top-N outcome; peak is their best positional finish."],
  ]],
  ["Draft day", [
    ["Taken checkboxes", "Stored in your browser only; survive refresh. Reset draft clears them."],
  ]],
  ["In-season", [
    ["Now score", "The same 1–100 scale, re-estimated for 2024–26 rookies with draft slot plus NFL production to date: points per elapsed week over replacement (missed weeks count as zero), target share, carry share. Trained by replaying the same week of the season for every 2013–23 class."],
    ["When NFL data is used", "Only at checkpoints where it beat draft slot out of class. Six weeks into the rookie year it helps RBs and WRs a lot (rank correlation 0.53→0.64 for WRs); rookie QBs and TEs lean on draft slot until later."],
    ["Change", "Now score minus the draft-slot score. Risers are buy-low checks; fallers are sell-before-the-market checks."],
  ]],
  ["League view", [
    ["Value", "Median of six market sources, repriced to this league's actual scoring (6pt pass TD, half-PPR, TE +0.25/rec) using multipliers computed from real NFL weeks scored both ways, then age-weighted."],
    ["×mult", "That repricing multiplier. >1 = this league values him more than the market convention does (QBs, TEs here); <1 = less."],
    ["Win-now", "Projected starter points per game over replacement — a lineup-strength meter, independent of dynasty value."],
    ["Trade matches / proposals", "Matches: teams strong where you're weak that need QB. Proposals: packages a market-value dashboard scores as fair for them while you gain league-adjusted value."],
  ]],
  ["Data & discipline", [
    ["Freshness", "Market values and the in-season view refresh daily at 11:00 UTC via CI. Draft-board inputs are frozen at draft time: college stats, combine, slot, age — no camp hype, ever."],
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
