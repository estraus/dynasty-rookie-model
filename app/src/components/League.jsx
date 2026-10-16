import React, { useEffect, useState } from "react";
import { POSITIONS } from "../lib.js";
import { PosTag, Tape } from "./ui.jsx";

const SOURCE_LABELS = {
  ktc: "KeepTradeCut", fc: "FantasyCalc", dp: "DynastyProcess",
  ds: "DynastySuperflex", adp: "DD ADP", dd: "DynastyDaddy",
};

// spread of a player's per-source values: min–max of the sources that price him
function spread(vals) {
  const priced = Object.values(vals || {}).filter(Boolean);
  if (priced.length < 2) return null;
  return [Math.min(...priced), Math.max(...priced)];
}

// League view: my roster vs league (Sleeper rosters, multi-source SF values).
// Reads league.json written by `python -m src.ingest.sleeper`; all logic lives
// in the pipeline, this renders the consensus (median) with per-source detail.
export default function League() {
  const [data, setData] = useState(null);
  const [team, setTeam] = useState(null);

  useEffect(() => {
    fetch("./league.json")
      .then((r) => (r.ok ? r.json() : null))
      .then(setData)
      .catch(() => setData(null));
  }, []);

  if (data === null)
    return (
      <div className="p-8 text-center font-mono text-xs text-mute">
        No league data. Run <code className="text-dim">python -m src.ingest.sleeper</code> to pull
        your Sleeper league.
      </div>
    );

  const { me, teams, ranks, totals, win_now, picks, matches, proposals, depth, sources } = data;
  const maxWin = win_now ? Math.max(...Object.values(win_now)) : 0;
  const n = Object.keys(teams).length;
  const owners = Object.keys(teams).sort((a, b) => totals[b] - totals[a]);
  const shown = team || me;

  return (
    <div className="space-y-7">
      <div className="flex items-baseline gap-3 flex-wrap">
        <h2 className="font-display font-bold text-xl tracking-wide uppercase">{data.league_name}</h2>
        <span className="font-mono text-[11px] text-mute tracking-wide">
          {data.season} · {sources?.length ?? "?"}-market median · repriced 6pt paTD ½PPR · picks:{" "}
          {picks.map((p) => `${p.round}.${String(p.slot).padStart(2, "0")}`).join(", ") || "none"}
        </span>
      </div>

      {/* my positional strength */}
      <div className="grid grid-cols-4 gap-2">
        {POSITIONS.map((pos) => {
          const r = ranks[me][pos];
          const tone =
            r <= n / 3 ? "text-up" : r > (2 * n) / 3 ? "text-down" : "text-brass";
          return (
            <div key={pos} className="border border-line bg-card rounded-sm p-3 text-center">
              <PosTag pos={pos} />
              <div className={`font-display font-bold text-2xl mt-1 tabular-nums ${tone}`}>
                {r}
                <span className="font-mono text-[11px] text-mute font-normal">/{n}</span>
              </div>
              <div className="font-mono text-[10px] text-mute tracking-wide">
                top-{depth[pos]} consensus rank
              </div>
            </div>
          );
        })}
      </div>

      {/* trade matches */}
      {matches.length > 0 && (
        <div>
          <div className="mb-2">
            <Tape note="they need QB, strong where you're weak">Trade matches</Tape>
          </div>
          <div className="space-y-1">
            {matches.map((m, i) => (
              <div
                key={i}
                className="flex items-center gap-3 px-3 py-2 rounded-sm border border-line bg-card text-sm"
              >
                <button
                  onClick={() => setTeam(m.owner)}
                  className="font-display font-semibold tracking-wide hover:text-brass"
                >
                  {m.owner}
                </button>
                <span className="font-mono text-[11px] text-mute tabular-nums">
                  QB #{m.qb_rank} · {m.pos} #{m.pos_rank}
                </span>
                <div className="flex-1 text-right font-mono text-[11px] text-dim truncate">
                  {m.pieces.map((p) => `${p.name} (${p.adj})`).join(" · ")}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* trade proposals: both dashboards say yes */}
      {proposals?.length > 0 && (
        <div>
          <div className="mb-2">
            <Tape note="fair on market value, you gain league-adjusted value">Proposals</Tape>
          </div>
          <div className="space-y-1">
            {proposals.map((p, i) => (
              <div
                key={i}
                className="flex items-center gap-3 px-3 py-2 rounded-sm border border-line bg-card text-sm"
              >
                <button
                  onClick={() => setTeam(p.owner)}
                  className="font-display font-semibold tracking-wide hover:text-brass shrink-0"
                >
                  {p.owner}
                </button>
                <span className="flex-1 font-mono text-[11px] truncate">
                  <span className="text-down/90">{p.give.join(" + ")}</span>
                  <span className="text-mute">{" → "}</span>
                  <span className="text-up/90">{p.get.join(" + ")}</span>
                </span>
                <span className="font-mono text-xs tabular-nums text-up shrink-0">+{p.my_net_adj}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-8">
        {/* league table */}
        <div>
          <div className="mb-2">
            <Tape note="dynasty value vs win-now (starter ppg over repl)">Teams</Tape>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="font-mono text-[10px] tracking-widest uppercase text-mute">
                <th className="text-left px-2 py-1 font-normal">team</th>
                <th className="text-right px-2 font-normal">value</th>
                <th className="text-right px-2 font-normal">win-now</th>
                {POSITIONS.map((p) => (
                  <th key={p} className="text-right px-2 font-normal">{p}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {owners.map((o) => {
                const wn = win_now?.[o];
                const wnTone =
                  wn != null && maxWin > 0 && wn >= 0.66 * maxWin
                    ? "text-up"
                    : wn != null && wn <= 0.33 * maxWin
                      ? "text-down"
                      : "text-dim";
                return (
                  <tr
                    key={o}
                    onClick={() => setTeam(o)}
                    className={`cursor-pointer hover:bg-raised ${o === shown ? "bg-raised" : ""} ${
                      o === me ? "text-brass" : ""
                    }`}
                  >
                    <td className="px-2 py-1 font-display font-semibold tracking-wide">{o}</td>
                    <td className="px-2 text-right font-mono text-xs tabular-nums">
                      {(totals[o] / 1000).toFixed(0)}k
                    </td>
                    <td className={`px-2 text-right font-mono text-xs tabular-nums ${wnTone}`}>
                      {wn != null ? wn.toFixed(0) : "—"}
                    </td>
                    {POSITIONS.map((p) => (
                      <td key={p} className="px-2 text-right font-mono text-xs tabular-nums text-dim">
                        {ranks[o][p]}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* selected roster: league-adjusted value, multiplier, market median */}
        <div>
          <div className="mb-2">
            <Tape note="league-adj · ×mult · market median">
              {shown === me ? "My roster" : `${shown}'s roster`}
            </Tape>
          </div>
          <div className="space-y-0.5 max-h-[28rem] overflow-y-auto pr-1">
            {teams[shown]
              .filter((p) => p.adj > 0)
              .map((p, i) => {
                const sp = spread(p.vals);
                return (
                  <div
                    key={i}
                    title={
                      Object.entries(p.vals || {})
                        .filter(([, v]) => v)
                        .map(([s, v]) => `${SOURCE_LABELS[s] || s}: ${v}`)
                        .join("\n") + (sp ? `\nspread: ${sp[0]}–${sp[1]}` : "")
                    }
                    className="flex items-center gap-2 px-2 py-1 rounded-sm text-sm hover:bg-raised"
                  >
                    <PosTag pos={p.pos} className="w-6" />
                    <span className="flex-1 font-display font-semibold tracking-wide truncate">
                      {p.name}
                      {p.trend > 150 && <span className="ml-1 text-xs text-up">▲</span>}
                      {p.trend < -150 && <span className="ml-1 text-xs text-down">▼</span>}
                    </span>
                    <span className="font-mono text-[11px] text-mute">{p.age}</span>
                    <span className="w-14 text-right font-mono text-xs tabular-nums text-ink">{p.adj}</span>
                    <span
                      className={`w-12 text-right font-mono text-[11px] tabular-nums ${
                        p.mult > 1 ? "text-up" : p.mult < 1 ? "text-down" : "text-mute"
                      }`}
                    >
                      {p.mult !== 1 ? `×${p.mult.toFixed(2)}` : "—"}
                    </span>
                    <span className="w-14 text-right font-mono text-[11px] tabular-nums text-mute">
                      {p.med}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      </div>
    </div>
  );
}
