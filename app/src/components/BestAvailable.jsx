import React from "react";
import { POSITIONS, scoreColor } from "../lib.js";
import { PosTag, Tape } from "./ui.jsx";

// draft-day panel: best available overall + per position among untaken players.
// xVOR = expected league ppg over replacement (best 2 of first 3 seasons).
// Cross-position order comes from the score, which also weighs career length.
export default function BestAvailable({ rows, taken, onSelect }) {
  const avail = rows.filter((r) => !taken.has(r.player_id));
  const overall = avail.slice(0, 5); // rows already sorted by score
  return (
    <div className="space-y-5 border border-line rounded-sm bg-card p-4">
      <Group title="Best available" players={overall} onSelect={onSelect} big />
      <div className="grid grid-cols-2 gap-4">
        {POSITIONS.map((pos) => (
          <Group
            key={pos}
            title={pos}
            players={avail.filter((r) => r.pos === pos).slice(0, 3)}
            onSelect={onSelect}
          />
        ))}
      </div>
      <p className="font-mono text-[10px] tracking-wide text-mute">
        {avail.length} of {rows.length} available · xVOR = expected league pts/wk over replacement,
        first 3 seasons
      </p>
    </div>
  );
}

function Group({ title, players, onSelect, big }) {
  return (
    <div>
      <div className="mb-2">
        <Tape>{title}</Tape>
      </div>
      <div className="space-y-0.5">
        {players.length === 0 && <p className="font-mono text-xs text-mute">— none —</p>}
        {players.map((r) => (
          <button
            key={r.player_id}
            onClick={() => onSelect(r)}
            className="w-full flex items-baseline gap-2 px-2 py-1 rounded-sm hover:bg-raised"
          >
            <span
              className={`font-display font-bold tabular-nums ${scoreColor(r.score_1_100)} ${
                big ? "text-xl w-9" : "w-7"
              } text-right`}
            >
              {r.score_1_100}
            </span>
            <span className="flex-1 text-left font-display font-semibold tracking-wide truncate">
              {r.name}
            </span>
            {r.exp_vor != null && (
              <span className="font-mono text-[11px] tabular-nums text-mute">{r.exp_vor.toFixed(1)}</span>
            )}
            <PosTag pos={r.pos} />
          </button>
        ))}
      </div>
    </div>
  );
}
