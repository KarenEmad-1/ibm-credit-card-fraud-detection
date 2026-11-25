// Check that the static site answers like the API: node scripts/check_static_site.mjs <site-dir> <api-base-url>
//
// Runs dashboard/js/static-backend.js in Node against the exported files and compares it
// with a running FastAPI server that has the same sample bundle. Exits non-zero on any
// difference. Numbers may differ by 1e-5 relative (summation order, rounded SHAP values).
import { readFile } from "node:fs/promises";
import { join } from "node:path";

const [site, baseArg] = process.argv.slice(2);
if (!site || !baseArg) { console.error("usage: check_static_site.mjs <site-dir> <api-base-url>"); process.exit(2); }
const base = baseArg.replace(/\/$/, "");

const source = await readFile(join(site, "js", "static-backend.js"));
const { createBackend } = await import(`data:text/javascript;base64,${source.toString("base64")}`);
const backend = createBackend(async (name) => {
  const b = await readFile(join(site, "data", name));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
});

const query = (params) => new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
async function api(path, params = {}) {
  const r = await fetch(`${base}${path}?${query(params)}`);
  if (!r.ok) throw new Error(`API ${path} -> ${r.status}`);
  return r.json();
}

/** First difference between two JSON values, or null. */
function diff(a, b, at = "") {
  if (typeof a === "number" && typeof b === "number") {
    return Math.abs(a - b) <= 1e-5 * Math.max(1, Math.abs(a), Math.abs(b)) ? null : `${at}: api ${a}, static ${b}`;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    if (Array.isArray(a) !== Array.isArray(b)) return `${at}: array vs object`;
    const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
    if (ka.join() !== kb.join()) return `${at}: keys ${ka.slice(0, 40)} vs ${kb.slice(0, 40)}`;
    for (const k of ka) { const d = diff(a[k], b[k], `${at}.${k}`); if (d) return d; }
    return null;
  }
  return a === b ? null : `${at}: api ${JSON.stringify(a)}, static ${JSON.stringify(b)}`;
}

let checks = 0;
const failures = [];
function record(label, d) { checks++; if (d) failures.push(`${label}${d}`); }
async function same(path, params = {}) {
  const [a, b] = await Promise.all([api(path, params), backend.handle(path, params)]);
  record(`${path}?${query(params)} `, diff(a, JSON.parse(JSON.stringify(b))));
  return a;
}

/** Read the API's event stream until `until(events)` is true. */
async function streamEvents(params, until) {
  const ctl = new AbortController();
  const r = await fetch(`${base}/api/replay?${query(params)}`, { signal: ctl.signal });
  const events = [];
  let text = "";
  const decoder = new TextDecoder();
  for await (const chunk of r.body) {
    text += decoder.decode(chunk, { stream: true });
    let end;
    while ((end = text.indexOf("\n\n")) >= 0) {
      const m = /^event: (\w+)\ndata: (.*)$/s.exec(text.slice(0, end));
      if (m) events.push([m[1], JSON.parse(m[2])]);
      text = text.slice(end + 2);
    }
    if (until(events)) break;
  }
  ctl.abort();
  return events;
}

// 1. Saved responses
for (const path of ["/api/splits", "/api/models", "/api/curves", "/api/drift", "/api/ablations"]) await same(path);
const { models } = await api("/api/models");
const days = (await same("/api/triage/days", { n: 14, seed: 7 })).days;

// 2. Computed in the browser, for every model
const keys = models.map((m) => m.key);
let cards = [];
for (const model of keys) {
  const e = encodeURIComponent(model);
  await same(`/api/segments/${e}`);
  await same(`/api/threshold/${e}`);
  for (const t of [0, 0.001, 0.1, 0.5, 0.9, 1]) await same(`/api/threshold/${e}`, { t });
  cards = (await same("/api/cards/attacks", { model, limit: 40 })).cards;
  for (const date of days) {
    await same("/api/triage", { date, model, k: 24 });
    await same("/api/triage/reveal", { date, model, k: 24 });
  }
}
const explained = new Set();
for (const [i, c] of cards.entries()) {
  const tl = await same(`/api/cards/${c.user}/${c.card}`, { model: keys[i % keys.length] });
  for (const r of [tl.rows[0], tl.rows[tl.rows.length - 1], tl.rows.find((x) => x.is_fraud)]) explained.add(r.txn_id);
}

// 3. Game: the draw differs by design, so check that every drawn round is valid and reveals the same
const seen = { fraud: 0, legit: 0 };
for (let seed = 0; seed < 60; seed++) {
  const model = keys[seed % keys.length];
  const round = await backend.handle("/api/game/round", { seed, model });
  const reveal = await same(`/api/game/reveal/${round.round}`);
  seen[reveal.is_fraud ? "fraud" : "legit"]++;
  const { transaction } = await api(`/api/transactions/${round.round}`);
  for (const f of ["amount", "merchant", "city", "user", "card"]) record(`round ${seed} ${f}`, diff(transaction[f], round.transaction[f]));
  explained.add(round.round);
}
record("game mix", seen.fraud && seen.legit ? null : `: ${JSON.stringify(seen)}`);

// 4. SHAP, including that it reproduces the saved score
const { explainable } = await api("/api/models");
for (const id of explained) await same(`/api/explain/${encodeURIComponent(explainable[0])}/${id}`);

// 5. Replay: the same events in the same order
for (const [model, speed, start, n] of [["CatBoost A", 4320, "2019-01-01T00:00:00", 12], ["LogReg B", 1440, "2019-01-10T06:30:00", 8],
  ["CatBoost B", 30, "2019-01-03T12:00:00", 6], ["CatBoost A", 4320, "2019-01-28T00:00:00", 1e9]]) {
  const params = { model, speed, start };
  const a = await streamEvents(params, (ev) => ev.length >= n || ev.at(-1)?.[0] === "end");
  const b = [];
  for await (const ev of backend.replayEvents(params)) { b.push(JSON.parse(JSON.stringify(ev))); if (b.length >= a.length) break; }
  record(`replay ${JSON.stringify(params)} `, a.length < 2 ? "too few events" : diff(a, b));
  if (n === 1e9) record("replay end ", a.at(-1)[0] === "end" ? null : "no end event");
}

// 6. Headline number, independent of the API comparison
const cb = (await backend.handle("/api/models")).models.find((m) => m.key === "CatBoost B");
record("CatBoost B PR-AUC ", cb.metrics.pr_auc.toFixed(4) === "0.4808" ? null : String(cb.metrics.pr_auc));

if (failures.length) {
  console.error(`${failures.length} of ${checks} checks failed:`);
  for (const f of failures.slice(0, 20)) console.error(`  ${f}`);
  process.exit(1);
}
console.log(`static site matches the API: ${checks} checks (${explained.size} SHAP explanations, game ${JSON.stringify(seen)})`);
process.exit(0);
