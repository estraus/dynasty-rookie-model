import React, { useEffect, useMemo, useState } from "react";
import { POSITIONS, POS_TEXT, useTaken } from "./lib.js";
import Board from "./components/Board.jsx";
import Drawer from "./components/Drawer.jsx";
import Delta from "./components/Delta.jsx";
import BestAvailable from "./components/BestAvailable.jsx";
import League from "./components/League.jsx";
import About from "./components/About.jsx";
import InSeason from "./components/InSeason.jsx";

const VIEWS = [
  ["board", "Board"],
  ["delta", "KTC Delta"],
  ["draft", "Draft Day"],
  ["inseason", "In-Season"],
  ["league", "League"],
  ["about", "About"],
];

export default function App() {
  const [data, setData] = useState(null);
  const [view, setView] = useState("board");
  const [search, setSearch] = useState("");
  const [posFilter, setPosFilter] = useState(new Set());
  const [tierFilter, setTierFilter] = useState(new Set());
  const [sortKey, setSortKey] = useState("score_1_100");
  const [selected, setSelected] = useState(null);
  const { taken, toggle, reset } = useTaken();

  useEffect(() => {
    fetch("./board.json")
      .then((r) => r.json())
      // model_rank = overall rank by score (board.json is score-sorted);
      // capital_rank = rank by draft slot alone — the anchor both model and
      // market deviate from (pure presentation, no model logic)
      .then((rows) => {
        const byPick = [...rows].sort((a, b) => a.pick - b.pick);
        const capRank = new Map(byPick.map((r, i) => [r.player_id, i + 1]));
        setData(
          rows.map((r, i) => ({
            ...r,
            model_rank: i + 1,
            capital_rank: capRank.get(r.player_id),
          }))
        );
      })
      .catch(() => setData([]));
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    let out = data.filter((r) => {
      if (posFilter.size && !posFilter.has(r.pos)) return false;
      if (tierFilter.size && !tierFilter.has(r.tier)) return false;
      if (search && !r.name.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
    const dir = sortKey === "pick" ? 1 : -1; // lower pick = better, everything else desc
    out = [...out].sort((a, b) => {
      const av = a[sortKey] ?? -Infinity;
      const bv = b[sortKey] ?? -Infinity;
      return (av - bv) * dir;
    });
    return out;
  }, [data, posFilter, tierFilter, search, sortKey]);

  if (!data) return <div className="p-10 font-mono text-xs tracking-widest uppercase text-mute">Loading board…</div>;
  if (data.length === 0)
    return <div className="p-10 font-mono text-xs text-down">Could not load board.json.</div>;

  return (
    <div className="min-h-screen">
      <header className="border-b border-line-strong px-5 pt-4 pb-3 sticky top-0 bg-board/95 backdrop-blur z-10">
        <div className="max-w-5xl mx-auto">
        <div className="flex items-baseline gap-5 flex-wrap">
          <h1 className="font-display font-bold text-2xl tracking-wide uppercase leading-none">
            The Board
            <span className="ml-3 font-mono text-[11px] font-normal normal-case tracking-widest text-mute align-middle">
              2026 CLASS · SF · ½PPR
            </span>
          </h1>
          <nav className="flex gap-5">
            {VIEWS.map(([k, label]) => (
              <button
                key={k}
                onClick={() => setView(k)}
                className={`font-display font-semibold text-sm uppercase tracking-[0.15em] pb-0.5 border-b-2 ${
                  view === k
                    ? "text-brass border-brass"
                    : "text-dim border-transparent hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="flex-1" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search player…"
            className="bg-well border border-line rounded-sm px-3 py-1 font-mono text-xs w-44 placeholder:text-mute focus:outline-none focus:border-brass/60"
          />
        </div>

        <div className="flex items-center gap-4 mt-3 flex-wrap">
          <FilterChips
            label="Pos"
            options={POSITIONS}
            active={posFilter}
            setActive={setPosFilter}
            styleMap={POS_TEXT}
          />
          <FilterChips
            label="Tier"
            options={[1, 2, 3, 4, 5]}
            active={tierFilter}
            setActive={setTierFilter}
          />
          <div className="flex-1" />
          <span className="font-mono text-[11px] text-mute">{taken.size} taken</span>
          {taken.size > 0 && (
            <button
              onClick={reset}
              className="font-mono text-[11px] uppercase tracking-wider text-dim hover:text-down"
            >
              reset draft
            </button>
          )}
        </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-5 py-5">
        {view === "board" && (
          <Board
            rows={rows}
            taken={taken}
            onToggle={toggle}
            onSelect={setSelected}
            sortKey={sortKey}
            setSortKey={setSortKey}
          />
        )}
        {view === "delta" && <Delta rows={rows} taken={taken} onSelect={setSelected} />}
        {view === "league" && <League />}
        {view === "inseason" && <InSeason />}
        {view === "about" && <About />}
        {view === "draft" && (
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h2 className="font-mono text-[11px] tracking-widest uppercase text-mute mb-3">
                Board — check to mark taken
              </h2>
              <Board
                rows={rows}
                taken={taken}
                onToggle={toggle}
                onSelect={setSelected}
                sortKey={sortKey}
                setSortKey={setSortKey}
              />
            </div>
            <div className="md:sticky md:top-32 md:self-start">
              <BestAvailable rows={data} taken={taken} onSelect={setSelected} />
            </div>
          </div>
        )}
      </main>

      <Drawer player={selected} onClose={() => setSelected(null)} />
    </div>
  );
}

function FilterChips({ label, options, active, setActive, styleMap }) {
  const toggle = (o) =>
    setActive((prev) => {
      const next = new Set(prev);
      next.has(o) ? next.delete(o) : next.add(o);
      return next;
    });
  return (
    <div className="flex items-center gap-1.5">
      <span className="font-mono text-[10px] tracking-widest uppercase text-mute">{label}</span>
      {options.map((o) => {
        const on = active.has(o);
        return (
          <button
            key={o}
            onClick={() => toggle(o)}
            className={`font-mono text-[11px] px-2 py-0.5 rounded-sm border ${
              on
                ? `border-line-strong bg-raised ${styleMap?.[o] || "text-brass"}`
                : "border-line text-dim hover:border-line-strong hover:text-ink"
            }`}
          >
            {label === "Tier" ? `T${o}` : o}
          </button>
        );
      })}
    </div>
  );
}
