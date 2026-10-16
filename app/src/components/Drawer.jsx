import React from "react";
import { scoreColor } from "../lib.js";
import { PosTag, Tape } from "./ui.jsx";

const STAT_LABELS = {
  rec: "Rec", rec_yds: "Rec Yds", rec_td: "Rec TD",
  rush_yds: "Rush Yds", rush_td: "Rush TD", pass_yds: "Pass Yds", pass_td: "Pass TD",
};

// player detail drawer: score, P(hit), drivers w/ direction, college line, slot, age
export default function Drawer({ player, onClose }) {
  if (!player) return null;
  const stats = Object.entries(player.college_stats || {}).filter(([, v]) => v != null && v !== 0);
  return (
    <>
      <div className="fixed inset-0 bg-black/60 z-20" onClick={onClose} />
      <aside className="fixed right-0 top-0 h-full w-full max-w-md bg-card border-l border-line-strong z-30 overflow-y-auto p-6">
        <button
          onClick={onClose}
          className="font-mono text-[11px] tracking-widest uppercase text-mute hover:text-ink mb-5"
        >
          ✕ close
        </button>

        <div className="flex items-baseline gap-3">
          <h2 className="font-display font-bold text-3xl tracking-wide uppercase">{player.name}</h2>
          <PosTag pos={player.pos} />
        </div>
        <p className="font-mono text-xs text-dim mt-1">
          {player.school} · pick {player.round}.{String(player.pick).padStart(2, "0")}
          {player.age ? ` · age ${player.age.toFixed?.(1) ?? player.age}` : ""}
        </p>

        <div className="flex gap-6 mt-6">
          <Stat label="Score" value={player.score_1_100} cls={scoreColor(player.score_1_100)} />
          <Stat label="P(hit)" value={`${Math.round(player.p_hit * 100)}%`} />
          <Stat label="Tier" value={player.tier} />
        </div>

        <Section title="Top drivers">
          <ul className="space-y-1.5">
            {player.drivers.map((d, i) => (
              <li key={i} className="text-sm">
                <div className="flex items-center justify-between">
                  <span>
                    <span className={d.direction === "positive" ? "text-up" : "text-down"}>
                      {d.direction === "positive" ? "▲" : "▼"}
                    </span>{" "}
                    {d.name}
                  </span>
                  <span className="font-mono text-xs text-mute tabular-nums">{d.magnitude.toFixed(2)}</span>
                </div>
                <div className="pl-4 font-mono text-[11px] text-mute">{driverDetail(d, player.pos)}</div>
              </li>
            ))}
          </ul>
        </Section>

        <Section title="Model lean vs draft capital">
          <p className="text-sm text-dim">{leanText(player.lean)}</p>
        </Section>

        {stats.length > 0 && (
          <Section title="Final college season">
            <div className="grid grid-cols-3 gap-2">
              {stats.map(([k, v]) => (
                <div key={k} className="bg-well border border-line rounded-sm px-2 py-1.5">
                  <div className="font-mono text-[10px] tracking-wider uppercase text-mute">
                    {STAT_LABELS[k] || k}
                  </div>
                  <div className="font-display font-semibold text-base tabular-nums">{v}</div>
                </div>
              ))}
            </div>
          </Section>
        )}

        {player.ktc_rank != null && <ModelVsMarket player={player} />}
        {player.comps?.length > 0 && <Comps player={player} />}
      </aside>
    </>
  );
}

// why model ≠ KTC: both deviate from the draft-capital anchor. model−capital is
// the production/age lean (the drivers above); KTC−capital is market sentiment.
function ModelVsMarket({ player }) {
  const { capital_rank: cap, model_rank: model, ktc_rank: ktc } = player;
  const modelVsCap = cap - model; // + = model boosts him past his slot
  const mktVsCap = cap - ktc; // + = market boosts him past his slot
  const rungs = [
    ["Draft capital", cap, null, "slot alone"],
    ["Model", model, modelVsCap, modelNote(player.lean, modelVsCap)],
    ["KTC market", ktc, mktVsCap, mktVsCap >= 0 ? "market hype vs slot" : "market cooler than slot"],
  ];
  return (
    <Section title="Model vs market">
      <div className="space-y-1">
        {rungs.map(([label, rank, dev, note]) => (
          <div key={label} className="flex items-baseline gap-2 text-sm">
            <span className="w-24 font-mono text-[11px] uppercase tracking-wider text-mute">{label}</span>
            <span className="w-10 text-right font-display font-bold tabular-nums">#{rank}</span>
            <span className={`w-10 text-right font-mono text-xs tabular-nums ${
              dev == null ? "text-mute" : dev >= 0 ? "text-up" : "text-down"
            }`}>
              {dev == null ? "—" : `${dev >= 0 ? "▲" : "▼"}${Math.abs(dev)}`}
            </span>
            <span className="font-mono text-[11px] text-mute">{note}</span>
          </div>
        ))}
      </div>
      <p className="text-sm text-dim mt-3">{verdict(player.delta_vs_ktc, player.lean, mktVsCap)}</p>
      {player.market_ranks && Object.keys(player.market_ranks).length > 0 && (
        <p className="font-mono text-[11px] text-mute mt-2 tabular-nums">
          other markets:{" "}
          {Object.entries(player.market_ranks)
            .map(([s, r]) => `${MARKET_LABELS[s] || s} #${r}`)
            .join(" · ")}
        </p>
      )}
    </Section>
  );
}

// specific reading of one driver: raw value + how far off the position's
// training norm (z). value null = stat missing, imputed at the pool mean-ish 0.
const FLAG_DRIVERS = new Set(["no combine", "early declare", "power conference"]);

function driverDetail(d, pos) {
  if (d.value == null) return "stat missing — imputed; the flag itself is doing the work";
  if (d.name === "draft capital")
    return d.z <= 0
      ? `picked earlier than the typical drafted ${pos}`
      : `picked later than the typical drafted ${pos}`;
  if (FLAG_DRIVERS.has(d.name)) return d.value >= 0.5 ? "yes" : "no";
  const mag = Math.abs(d.z);
  const side = d.z >= 0 ? "above" : "below";
  const grade = mag >= 1.5 ? "far " : mag >= 0.5 ? "" : "near the";
  return `${d.value} · z ${d.z >= 0 ? "+" : ""}${d.z} — ${
    mag < 0.5 ? `near the ${pos} norm` : `${grade}${side} the ${pos} norm`
  }`;
}

const MARKET_LABELS = { dp: "DynastyProcess", ds: "DynastySF", adp: "DD ADP", dd: "DynastyDaddy" };

function Comps({ player }) {
  return (
    <Section title="Closest historical profiles">
      <div className="space-y-1.5">
        {player.comps.map((c, i) => (
          <div key={i} className="flex items-baseline gap-2 text-sm">
            <span className="font-display font-semibold tracking-wide">{c.name}</span>
            <span className="font-mono text-[11px] text-mute tabular-nums">
              {c.year} · pk {c.pick}
            </span>
            <span className={`font-mono text-[11px] font-semibold ${c.hit ? "text-up" : "text-down"}`}>
              {c.hit ? "HIT" : "MISS"}
            </span>
            {c.peak != null && (
              <span className="font-mono text-[11px] text-mute">peak {player.pos}{c.peak}</span>
            )}
            <span className="flex-1 text-right font-mono text-[11px] text-mute truncate">
              {c.why.join(" · ")}
            </span>
          </div>
        ))}
      </div>
      <p className="font-mono text-[10px] text-mute mt-2">
        nearest neighbors in model feature space, weighted by what the model weighs ·
        "why" = the weightiest features where the profiles match
      </p>
    </Section>
  );
}

// the model's rank move vs capital has two possible engines: the production/age
// lean (the drivers), or positional hit rates at that slot. lean tells which.
function modelNote(lean, dev) {
  if (lean != null && lean >= 0.03) return "production/age drivers boost him";
  if (lean != null && lean <= -0.03)
    return dev > 0 ? "slot hit-rate carries him despite cooler drivers" : "production/age drivers cool on him";
  return "tracks draft capital";
}

function verdict(delta, lean, mktVsCap) {
  const d = Math.round(delta);
  if (Math.abs(d) <= 2) return "Model and market roughly agree — no edge either way.";
  const modelSide =
    lean != null && lean >= 0.03
      ? "the model's production/age drivers like him more than his slot"
      : lean != null && lean <= -0.03
        ? "the model is capital-driven here (drivers are cooler)"
        : "the model tracks his draft capital";
  const marketSide =
    mktVsCap >= 3
      ? "the market pays a premium over his slot"
      : mktVsCap <= -3
        ? "the market is cooler than his slot"
        : "the market roughly prices his slot";
  return `${d > 0 ? "+" : ""}${d} vs KTC — ${d > 0 ? "target" : "fade"}: ${modelSide}, while ${marketSide}.`;
}

function Stat({ label, value, cls = "" }) {
  return (
    <div className="flex flex-col">
      <span className="font-mono text-[10px] tracking-widest uppercase text-mute">{label}</span>
      <span className={`font-display font-bold text-3xl tabular-nums mt-0.5 ${cls}`}>{value}</span>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="mt-7">
      <div className="mb-2">
        <Tape>{title}</Tape>
      </div>
      {children}
    </div>
  );
}

function leanText(lean) {
  if (lean == null) return "No production model available.";
  if (Math.abs(lean) < 0.03) return "Production model agrees with draft capital.";
  return lean > 0
    ? `Production/age model likes him more than his draft slot (+${lean.toFixed(2)}). Possible value.`
    : `Production/age model is cooler than his draft slot (${lean.toFixed(2)}). Capital-driven.`;
}
