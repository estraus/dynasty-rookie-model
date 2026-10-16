import React, { useEffect, useMemo, useState } from "react";
import { POS_TICK, scoreColor, signed } from "../lib.js";
import { PosTag } from "./ui.jsx";

// In-season re-valuation of rookies still inside their 3-season window.
// Reads inseason.json (python -m src.model.inseason apply, refreshed daily).
// Owner comes from league.json -- a presentation join by name, no model logic.
const SORTS = [
  ["now_score", "Now"],
  ["change", "Change"],
  ["draft_score", "At draft"],
];

export default function InSeason({ onSelectName }) {
  const [data, setData] = useState(null);
  const [owners, setOwners] = useState({ map: {}, me: null });
  const [cls, setCls] = useState(null);
  const [sortKey, setSortKey] = useState("now_score");

  useEffect(() => {
    fetch("./inseason.json")
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null));
    fetch("./league.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((lg) => {
        if (!lg) return;
        const map = {};
        for (const [owner, roster] of Object.entries(lg.teams || {}))
          for (const p of roster) map[p.name] = owner;
        setOwners({ map, me: lg.me });
      })
      .catch(() => {});
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    const out = data.players.filter((p) => cls == null || p.class === cls);
    return [...out].sort((a, b) => b[sortKey] - a[sortKey]);
  }, [data, cls, sortKey]);

  if (data === null)
    return (
      <div className="p-8 text-center font-mono text-xs text-mute">
        No in-season data. Run <code className="text-dim">python -m src.model.inseason apply</code>.
      </div>
    );

  const classes = [...new Set(data.players.map((p) => p.class))].sort((a, b) => b - a);
  return (
    <div className="space-y-3">
      <div className="flex items-baseline gap-3 flex-wrap">
        <h2 className="font-display font-bold text-xl tracking-wide uppercase">In-season</h2>
        <span className="font-mono text-[11px] text-mute tracking-wide">
          {data.season} · through week {data.week} · draft slot + NFL production to date
        </span>
      </div>
      <p className="text-sm text-dim max-w-2xl">
        The same 1–100 scale as the board, re-estimated with what each rookie has done in the NFL:
        points per week over replacement (missed weeks count as zero), target share, and carry
        share. <span className="text-ink">Change</span> is now minus the draft-slot score. Rising
        players are buy-low checks; falling ones are sell-before-the-market checks.
      </p>

      <div className="flex items-center gap-4 flex-wrap font-mono text-[11px]">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] tracking-widest uppercase text-mute">Class</span>
          {[null, ...classes].map((c) => (
            <button
              key={String(c)}
              onClick={() => setCls(c)}
              className={`px-2 py-0.5 rounded-sm border ${
                cls === c ? "border-line-strong bg-raised text-brass" : "border-line text-dim hover:text-ink"
              }`}
            >
              {c ?? "All"}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] tracking-widest uppercase text-mute">Sort</span>
          {SORTS.map(([k, label]) => (
            <button
              key={k}
              onClick={() => setSortKey(k)}
              className={`px-2 py-0.5 rounded-sm border ${
                sortKey === k ? "border-line-strong bg-raised text-brass" : "border-line text-dim hover:text-ink"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-3 px-3 pt-2 font-mono text-[10px] tracking-widest uppercase text-mute">
        <span className="flex-1">Player</span>
        <span className="w-12 text-right">Now</span>
        <span className="w-12 text-right">Chg</span>
        <span className="w-36 text-right hidden sm:block">Usage</span>
        <span className="w-16 text-right">Pts/g</span>
        <span className="w-24 text-right hidden md:block">Owner</span>
      </div>
      <div className="space-y-1">
        {rows.map((p) => {
          const owner = owners.map[p.name];
          const mine = owner && owner === owners.me;
          return (
            <div
              key={p.player_id}
              className={`relative flex items-center gap-3 pl-4 pr-3 py-1.5 rounded-sm border bg-card overflow-hidden ${
                mine ? "border-brass/40" : "border-line"
              }`}
            >
              <span className={`absolute left-0 top-0 bottom-0 w-[3px] ${POS_TICK[p.pos]}`} />
              <span className="flex-1 flex items-baseline gap-2 min-w-0">
                <span className="font-display font-semibold text-base tracking-wide truncate">{p.name}</span>
                <PosTag pos={p.pos} />
                <span className="font-mono text-[11px] text-mute whitespace-nowrap">
                  '{String(p.class).slice(2)} · pk {p.pick}
                </span>
              </span>
              <span className={`w-12 text-right font-display font-bold text-lg tabular-nums ${scoreColor(p.now_score)}`}>
                {p.now_score}
              </span>
              <span
                className={`w-12 text-right font-mono text-xs tabular-nums ${
                  p.change >= 5 ? "text-up" : p.change <= -5 ? "text-down" : "text-mute"
                }`}
                title={`At draft: ${p.draft_score}`}
              >
                {signed(p.change)}
              </span>
              <span className="w-36 text-right font-mono text-[11px] text-dim tabular-nums whitespace-nowrap hidden sm:block">
                {usage(p)}
              </span>
              <span className="w-16 text-right font-mono text-xs text-dim tabular-nums">
                {p.games ? p.ppg.toFixed(1) : "—"}
                <span className="block text-[10px] text-mute">{p.games} g</span>
              </span>
              <span
                className={`w-24 text-right font-mono text-[11px] truncate hidden md:block ${
                  mine ? "text-brass" : "text-mute"
                }`}
              >
                {owner ? (mine ? "you" : owner) : "FA"}
              </span>
            </div>
          );
        })}
      </div>
      <p className="font-mono text-[10px] text-mute pt-2">
        NFL usage only enters at a checkpoint where it beat draft slot alone in backtest; elsewhere
        (TEs at the end of year 1) the score stays on draft slot. Owner "FA" = not rostered in your league.
      </p>
    </div>
  );
}

function usage(p) {
  const parts = [];
  if (p.tgt_share >= 0.03) parts.push(`${Math.round(p.tgt_share * 100)}% tgt`);
  if (p.carry_share >= 0.03) parts.push(`${Math.round(p.carry_share * 100)}% car`);
  return parts.join(" · ") || "—";
}
