import React from "react";
import { scoreColor, signed } from "../lib.js";
import { PosTag } from "./ui.jsx";

// model rank vs KTC rank, sorted by gap. positive = model higher than KTC (target);
// negative = model lower (fade at cost).
export default function Delta({ rows, taken, onSelect }) {
  const withKtc = rows.filter((r) => r.delta_vs_ktc != null);
  if (withKtc.length === 0) {
    return (
      <div className="p-8 text-center font-mono text-xs text-mute">
        No KTC data loaded. Add <code className="text-dim">overrides/ktc_2026.csv</code>{" "}
        (name,ktc_rank) and re-run <code className="text-dim">src.model.score</code> to populate
        the delta view.
      </div>
    );
  }
  const sorted = [...withKtc].sort((a, b) => b.delta_vs_ktc - a.delta_vs_ktc);
  return (
    <div className="space-y-1">
      <p className="font-mono text-[11px] text-mute px-2 pb-2 tracking-wide">
        <span className="text-up">positive</span> = model over market (targets) ·{" "}
        <span className="text-down">negative</span> = model under (fades at cost)
      </p>
      {sorted.map((r) => (
        <button
          key={r.player_id}
          onClick={() => onSelect(r)}
          className={`w-full flex items-center gap-3 px-3 py-1.5 rounded-sm border border-line bg-card hover:bg-raised ${
            taken.has(r.player_id) ? "opacity-40" : ""
          }`}
        >
          <span
            className={`w-14 text-right font-display font-bold text-lg tabular-nums ${
              r.delta_vs_ktc > 0 ? "text-up" : "text-down"
            }`}
          >
            {signed(r.delta_vs_ktc)}
          </span>
          <span className="flex-1 flex items-baseline gap-2 text-left min-w-0">
            <span className="font-display font-semibold text-base tracking-wide truncate">{r.name}</span>
            <PosTag pos={r.pos} />
          </span>
          <span className="font-mono text-[11px] text-mute tabular-nums">
            cap #{r.capital_rank} · model #{r.model_rank ?? "—"} · KTC #{r.ktc_rank}
          </span>
          <span className={`w-8 text-right font-display font-bold tabular-nums ${scoreColor(r.score_1_100)}`}>
            {r.score_1_100}
          </span>
        </button>
      ))}
    </div>
  );
}
