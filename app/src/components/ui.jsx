import React from "react";
import { POS_TEXT } from "../lib.js";

// mono position tag in the magnet color — the only place position color appears in text
export function PosTag({ pos, className = "" }) {
  return (
    <span className={`font-mono text-[10px] font-semibold tracking-widest ${POS_TEXT[pos]} ${className}`}>
      {pos}
    </span>
  );
}

// tape rule: section header as a strip of label + hairline, war-room masking tape
export function Tape({ children, note }) {
  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-[11px] font-semibold tracking-[0.2em] uppercase text-dim whitespace-nowrap">
        {children}
      </span>
      <span className="flex-1 border-t border-line" />
      {note && (
        <span className="font-mono text-[10px] tracking-wider text-mute whitespace-nowrap">{note}</span>
      )}
    </div>
  );
}
