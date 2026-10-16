import React from "react";
import { LEAN_MIN, POS_TICK, confidence, fmtRange, scoreColor, signed } from "../lib.js";
import { PosTag, Tape } from "./ui.jsx";

const COLS = [
  ["score_1_100", "Score", "w-12"],
  ["exp_vor", "xVOR", "w-20"],
  ["pick", "Slot", "w-14"],
  ["lean", "Lean", "w-16"],
  ["delta_vs_ktc", "KTC Δ", "w-12"],
];

// board as a column of name cards; tier breaks are tape rules that print the cliff size
export default function Board({ rows, taken, onToggle, onSelect, sortKey, setSortKey }) {
  let lastTier = null;
  let prevScore = null;
  return (
    <div>
      <div className="flex items-center gap-3 px-3 pb-2 font-mono text-[10px] tracking-widest uppercase text-mute">
        <span className="w-4" />
        <span className="flex-1">Player</span>
        {COLS.map(([k, label, w]) => (
          <button
            key={k}
            onClick={() => setSortKey(k)}
            className={`${w} text-right cursor-pointer hover:text-ink ${
              sortKey === k ? "text-brass" : ""
            }`}
          >
            {label}
            {sortKey === k ? " ↓" : ""}
          </button>
        ))}
      </div>

      <div className="space-y-1">
        {rows.map((r) => {
          const isTaken = taken.has(r.player_id);
          const tierBreak = r.tier !== lastTier;
          const cliff =
            tierBreak && prevScore != null && sortKey === "score_1_100"
              ? prevScore - r.score_1_100
              : null;
          lastTier = r.tier;
          prevScore = r.score_1_100;
          return (
            <React.Fragment key={r.player_id}>
              {tierBreak && (
                <div className="pt-4 pb-1">
                  <Tape note={cliff != null ? `cliff −${cliff}` : null}>Tier {r.tier}</Tape>
                </div>
              )}
              <div
                className={`relative flex items-center gap-3 pl-4 pr-3 py-1.5 rounded-sm border border-line bg-card hover:bg-raised overflow-hidden ${
                  isTaken ? "opacity-40" : ""
                }`}
              >
                <span className={`absolute left-0 top-0 bottom-0 w-[3px] ${POS_TICK[r.pos]}`} />
                <input
                  type="checkbox"
                  checked={isTaken}
                  onChange={() => onToggle(r.player_id)}
                  className="accent-brass cursor-pointer"
                  title="Mark taken"
                />
                <button onClick={() => onSelect(r)} className="flex-1 flex items-baseline gap-2 text-left min-w-0">
                  <span
                    className={`font-display font-semibold text-base tracking-wide truncate ${
                      isTaken ? "line-through" : ""
                    }`}
                  >
                    {r.name}
                  </span>
                  <PosTag pos={r.pos} />
                  <span className="font-mono text-[11px] text-mute truncate">{r.school}</span>
                </button>
                <span
                  className={`w-12 text-right font-display font-bold text-lg tabular-nums ${scoreColor(r.score_1_100)}`}
                  title={`Confidence at ${r.pos}: ${confidence(r.reliability).label} (backtest rank correlation ${r.reliability?.toFixed(2) ?? "—"})`}
                >
                  {r.score_1_100}
                  {confidence(r.reliability).label === "low" && (
                    <sup className="text-mute text-[9px] font-mono font-normal">±</sup>
                  )}
                </span>
                <span
                  className="w-20 text-right font-mono text-xs tabular-nums leading-tight"
                  title="Expected league pts/wk over replacement (best 2 of first 3 seasons), with the 80% range of similar past players"
                >
                  <span className="text-ink">{r.exp_vor?.toFixed(1) ?? "—"}</span>
                  <span className="block text-[10px] text-mute">{fmtRange(r.value_range)}</span>
                </span>
                <span className="w-14 text-right font-mono text-xs text-dim tabular-nums">
                  {r.round}.{String(r.pick).padStart(2, "0")}
                </span>
                <span className="w-16 text-right font-mono text-xs tabular-nums">
                  <Lean v={r.lean} validated={r.lean_validated} />
                </span>
                <span
                  className="w-12 text-right font-mono text-xs text-dim tabular-nums"
                  title={r.delta_vs_ktc == null ? "No market price: KTC doesn't rank him yet" : undefined}
                >
                  {r.delta_vs_ktc == null ? <span className="text-mute text-[10px]">no price</span> : signed(r.delta_vs_ktc)}
                </span>
              </div>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

// lean = production/age adjustment to the draft-slot value, in pts/wk.
// Unvalidated (it didn't beat draft slot in backtest) -> rendered faint.
function Lean({ v, validated }) {
  if (v == null || Math.abs(v) < LEAN_MIN) return <span className="text-mute">·</span>;
  const up = v >= 0;
  const tone = validated ? (up ? "text-up" : "text-down") : (up ? "text-up/50" : "text-down/50");
  return (
    <span
      className={tone}
      title={validated ? undefined : "Unvalidated: production stats didn't beat draft slot in backtest"}
    >
      {up ? "▲" : "▼"} {Math.abs(v).toFixed(1)}
      {!validated && <sup className="text-mute">?</sup>}
    </span>
  );
}
