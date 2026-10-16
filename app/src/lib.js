import { useState, useEffect } from "react";

export const POSITIONS = ["QB", "RB", "WR", "TE"];

// magnet color per position: edge tick + text tag, never a filled pill
export const POS_TICK = { QB: "bg-qb", RB: "bg-rb", WR: "bg-wr", TE: "bg-te" };
export const POS_TEXT = { QB: "text-qb", RB: "text-rb", WR: "text-wr", TE: "text-te" };

// single-hue score ramp: brass for elite, then paper fades to mute
export function scoreColor(s) {
  if (s >= 88) return "text-brass";
  if (s >= 74) return "text-ink";
  if (s >= 58) return "text-dim";
  return "text-mute";
}

export function signed(n) {
  const v = Math.round(n);
  return v > 0 ? `+${v}` : `${v}`;
}

// persist a Set to localStorage (draft-day taken state survives refresh)
export function useTaken() {
  const [taken, setTaken] = useState(() => {
    try {
      return new Set(JSON.parse(localStorage.getItem("taken2026") || "[]"));
    } catch {
      return new Set();
    }
  });
  useEffect(() => {
    localStorage.setItem("taken2026", JSON.stringify([...taken]));
  }, [taken]);
  const toggle = (id) =>
    setTaken((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  const reset = () => setTaken(new Set());
  return { taken, toggle, reset };
}
