// In-browser replacement for the FastAPI backend, used by the GitHub Pages demo.
//
// GitHub Pages only serves files, so scripts/export_static_site.py saves the sample
// bundle's columns and the results that need Python (PR curves, segments, SHAP) under
// data/<split>/ (test, and val when exported). This module answers the same requests as api/routers/*.py from those files,
// so the screens do not know which backend they talk to. CI compares the two
// (scripts/check_static_site.mjs).
//
// Two deliberate differences from the API:
//   - SHAP is precomputed, so only a chosen set of transactions can be explained
//   - /api/game/round draws from that set, with a JavaScript random generator
// No DOM access here, so Node can run it too.

const TICK_SECONDS = 0.5;
const MAX_ALERTS_PER_TICK = 40;
const TAPE_ROWS = 10;
const DAY = 86400;
const SHOW = ["txn_id", "user", "card", "amount", "merchant", "city", "state", "mcc", "channel", "errors"];
const ALERT_COLS = ["txn_id", "user", "card", "amount", "merchant", "city", "state", "channel", "errors", "is_fraud"];
const TAPE_COLS = ["txn_id", "user", "card", "amount", "city", "state", "channel"];
const ARRAYS = { i4: Int32Array, i2: Int16Array, u2: Uint16Array, i1: Int8Array, u1: Uint8Array, f4: Float32Array };

const ATTEMPTS = 3;
const ATTEMPT_MS = 30000;

/** Download a data file. A dropped or stalled connection is retried. */
async function fetchFile(name) {
  for (let i = 1; ; i++) {
    try {
      const r = await fetch(new URL(`../data/${name}`, import.meta.url), { signal: AbortSignal.timeout(ATTEMPT_MS) });
      if (!r.ok) throw new Error(`${r.status}: data/${name}`);
      return await r.arrayBuffer();
    } catch (e) {
      if (i >= ATTEMPTS) throw e;
      await new Promise((done) => setTimeout(done, 500 * i));
    }
  }
}

/** Gunzip, unless the server already did (some send .gz with Content-Encoding). */
async function gunzip(buf) {
  const b = new Uint8Array(buf);
  if (b[0] !== 0x1f || b[1] !== 0x8b) return buf;
  return new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

const isoDay = (s) => new Date(s * 1000).toISOString().slice(0, 10);
const utc = (iso) => Math.floor(Date.parse(iso.length > 10 ? `${iso}Z` : `${iso}T00:00:00Z`) / 1000);

/** First index with a[i] >= v (numpy searchsorted, side="left"). */
function lowerBound(a, v) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; }
  return lo;
}

/** Small seeded generator (mulberry32): the same seed always gives the same round. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class ApiError extends Error {
  constructor(status, detail) { super(`${status}: ${detail}`); this.status = status; }
}

/** One split's backend: answers requests from the files in data/<split>/. */
function createSplitBackend(load) {
  const cache = new Map();
  // A failed download is forgotten, so the next request tries again
  const once = (key, fn) => {
    if (!cache.has(key)) cache.set(key, fn().catch((e) => { cache.delete(key); throw e; }));
    return cache.get(key);
  };
  const json = (name) => once(name, async () => JSON.parse(new TextDecoder().decode(await load(name))));
  const meta = () => json("static.json");

  const core = () => once("core", async () => {
    const [m, buf] = await Promise.all([meta(), load("core.bin.gz").then(gunzip)]);
    const c = { n: m.n_rows, dict: m.dictionaries };
    for (const { name, type, offset } of m.columns) c[name] = new ARRAYS[type](buf, offset, m.n_rows);
    c.rowOf = new Map();
    for (let i = 0; i < c.n; i++) c.rowOf.set(c.txn_id[i], i);
    return c;
  });
  const scores = (key) => once(`score:${key}`, async () => {
    const i = (await meta()).models.models.findIndex((x) => x.key === key);
    if (i < 0) throw new ApiError(404, `unknown model '${key}'`);
    return new Float32Array(await gunzip(await load(`score-${i}.bin.gz`)));
  });
  const modelMeta = async (key) => {
    const { models } = (await meta()).models;
    const m = key == null ? models.find((x) => x.key.startsWith("CatBoost")) ?? models[0] : models.find((x) => x.key === key);
    if (!m) throw new ApiError(404, `unknown model '${key}'`);
    return m;
  };

  const value = (c, name, i) => (c.dict[name] ? c.dict[name][c[name][i]] : c[name][i]);
  const record = (c, i, cols, extra) => {
    const r = {};
    for (const name of cols) r[name] = value(c, name, i);
    r.ts = c.ts[i];
    return Object.assign(r, extra);
  };
  const cardRows = (c, user, card) => {
    const rows = [];
    for (let i = 0; i < c.n; i++) if (c.user[i] === user && c.card[i] === card) rows.push(i);
    if (!rows.length) throw new ApiError(404, "card not found");
    return rows;
  };
  const rowOf = (c, txn) => {
    const i = c.rowOf.get(txn);
    if (i === undefined) throw new ApiError(404, "unknown transaction");
    return i;
  };

  async function explain(key, txn) {
    const m = await meta();
    if (key !== m.shap.model) throw new ApiError(400, `SHAP explanations are precomputed for ${m.shap.model} only`);
    const shard = await json(`shap/${txn % m.shap.shards}.json`);
    if (!shard[txn]) throw new ApiError(404, "no precomputed explanation for this transaction");
    const [base, logit, count, rest, items] = shard[txn];
    return {
      model: key, txn_id: txn, base_logit: base, logit, probability: 1 / (1 + Math.exp(-logit)),
      contributions: items.map(([f, v, s]) => ({ feature: m.shap.features[f], value: v, shap: s })),
      other_features: { count, shap: rest },
    };
  }

  // ------------------------------------------------------------ requests, as in api/routers
  async function threshold(key, p) {
    const [c, s, mm, m] = await Promise.all([core(), scores(key), modelMeta(key), meta()]);
    const t = p.t == null ? mm.threshold : +p.t;
    let alerts = 0, tp = 0, caught = 0, frauds = 0, total = 0;
    for (let i = 0; i < c.n; i++) {
      const hit = s[i] >= t;
      if (hit) alerts++;
      if (c.is_fraud[i]) {
        const a = Math.abs(c.amount[i]);
        frauds++; total += a;
        if (hit) { tp++; caught += a; }
      }
    }
    return {
      model: key, val_threshold: mm.threshold, threshold: t,
      alerts, alerts_per_day: alerts / m.models.n_days, alert_rate: alerts / c.n,
      tp, fp: alerts - tp, fn: frauds - tp, tn: c.n - alerts - (frauds - tp),
      precision: alerts ? tp / alerts : 0, recall: frauds ? tp / frauds : 0,
      fraud_amount_caught: caught, fraud_amount_total: total,
    };
  }

  async function gameRound(p) {
    const [c, m, mm] = await Promise.all([core(), meta(), modelMeta(p.model)]);
    const rand = rng(+(p.seed ?? 0));
    const r = rand();
    // Same mix as the API: 45% frauds, 30% ordinary legit, 25% the model's false alerts
    let pool = r < 0.45 ? m.game.fraud : r < 0.75 ? m.game.legit : m.game.false_alerts[mm.key];
    if (!pool.length) pool = m.game.legit;
    const i = rowOf(c, pool[Math.floor(rand() * pool.length)]);
    const rows = cardRows(c, c.user[i], c.card[i]);
    const pos = rows.indexOf(i);
    return {
      round: c.txn_id[i], transaction: record(c, i, SHOW),
      history: rows.slice(Math.max(0, pos - 6), pos).reverse().map((j) => record(c, j, SHOW)),
    };
  }

  async function gameReveal(txn) {
    const [c, m] = await Promise.all([core(), meta()]);
    const i = rowOf(c, txn);
    const all = await Promise.all(m.models.models.map(async (x) => {
      const score = (await scores(x.key))[i];
      return { model: x.key, score, threshold: x.threshold, flagged: score >= x.threshold };
    }));
    const e = await explain(m.shap.model, txn);
    return { txn_id: txn, is_fraud: c.is_fraud[i], scores: all, reasons: { model: e.model, items: e.contributions.slice(0, 4) } };
  }

  async function cardTimeline(user, card, p) {
    const mm = await modelMeta(p.model);
    const [c, s] = await Promise.all([core(), scores(mm.key)]);
    const rows = cardRows(c, user, card);
    const fts = rows.filter((i) => c.is_fraud[i]).map((i) => c.ts[i]);      // rows are in time order
    return {
      model: mm.key, threshold: mm.threshold, user, card,
      rows: rows.map((i) => record(c, i, SHOW, { score: s[i], is_fraud: c.is_fraud[i] })),
      attack_window: fts.length ? [fts[0] - 21 * DAY, fts[fts.length - 1] + 10 * DAY] : null,
    };
  }

  async function triageDay(p) {
    const mm = await modelMeta(p.model);
    const [c, s] = await Promise.all([core(), scores(mm.key)]);
    const day = utc(p.date);
    const lo = lowerBound(c.ts, day), hi = lowerBound(c.ts, day + DAY);
    if (hi <= lo) throw new ApiError(404, "no transactions that day");
    const q = Array.from({ length: hi - lo }, (_, j) => lo + j)
      .sort((a, b) => s[b] - s[a] || a - b).slice(0, +(p.k ?? 24));
    return { mm, c, s, lo, hi, q };
  }
  async function triage(p) {
    const { mm, c, s, lo, hi, q } = await triageDay(p);
    return {
      model: mm.key, date: p.date, threshold: mm.threshold, day_transactions: hi - lo,
      queue: q.map((i) => record(c, i, SHOW, { score: s[i] })),
    };
  }
  async function triageReveal(p) {
    const { c, lo, hi, q } = await triageDay(p);
    let frauds = 0, amount = 0;
    for (let i = lo; i < hi; i++) if (c.is_fraud[i]) { frauds++; amount += Math.abs(c.amount[i]); }
    return { labels: Object.fromEntries(q.map((i) => [c.txn_id[i], c.is_fraud[i]])), day_frauds: frauds, day_fraud_amount: amount };
  }

  async function handle(path, params = {}) {
    const p = Object.fromEntries(Object.entries(params).filter(([, v]) => v != null));
    const [a, b, d, ...more] = path.replace(/^\/api\//, "").split("/").map(decodeURIComponent);
    if (!more.length) {
      if (["models", "curves", "drift", "ablations"].includes(a) && b === undefined) return (await meta())[a];
      if (a === "threshold" && b) return threshold(b, p);
      if (a === "segments" && b) { const mm = await modelMeta(b); return (await meta()).segments[mm.key]; }
      if (a === "game" && b === "round") return gameRound(p);
      if (a === "game" && b === "reveal") return gameReveal(+d);
      if (a === "cards" && b === "attacks") { const mm = await modelMeta(p.model); return (await meta()).attacks[mm.key]; }
      if (a === "cards") return cardTimeline(+b, +d, p);
      if (a === "triage" && b === undefined) return triage(p);
      if (a === "triage" && b === "reveal") return triageReveal(p);
      if (a === "triage" && b === "days") return (await meta()).triage_days;
      if (a === "explain") return explain(b, +d);
    }
    throw new ApiError(404, `${path} is not available in the static demo`);
  }

  // ------------------------------------------------------------ replay
  /** The event sequence of GET /api/replay, as [event, data] pairs. */
  async function* replayEvents(p) {
    const mm = await modelMeta(p.model);
    const [c, s, m] = await Promise.all([core(), scores(mm.key), meta()]);
    const t = p.t == null ? mm.threshold : +p.t;
    const speed = +(p.speed ?? 60), step = Math.trunc(speed * 60 * TICK_SECONDS);
    const ts = c.ts;
    let sim = Math.max(p.start == null ? ts[0] : utc(p.start), ts[0]);
    let cursor = lowerBound(ts, sim), day = sim - (sim % DAY);
    const summary = (start) => {
      const lo = lowerBound(ts, start), hi = lowerBound(ts, start + DAY);
      let alerts = 0, tp = 0, frauds = 0, amount = 0;
      for (let i = lo; i < hi; i++) {
        const f = s[i] >= t;
        if (f) alerts++;
        if (c.is_fraud[i]) { frauds++; amount += Math.abs(c.amount[i]); if (f) tp++; }
      }
      const date = isoDay(start);
      return {
        date, transactions: hi - lo, alerts, true_alerts: tp, false_alerts: alerts - tp, frauds, missed: frauds - tp,
        fraud_amount: amount, psi: m.drift.daily[date] ?? {},
      };
    };
    yield ["start", { model: mm.key, threshold: t, split: m.models.split, sim_time: sim, speed, end_time: ts[c.n - 1] }];
    while (cursor < c.n) {
      const nxt = sim + step, hi = lowerBound(ts, nxt), n = hi - cursor;
      const alertIdx = [];
      let amount = 0, frauds = 0, trueAlerts = 0;
      for (let i = cursor; i < hi; i++) {
        amount += Math.abs(c.amount[i]);
        frauds += c.is_fraud[i];
        if (s[i] >= t) { alertIdx.push(i); trueAlerts += c.is_fraud[i]; }
      }
      const shown = alertIdx.slice(0, MAX_ALERTS_PER_TICK);
      // an even sample of all traffic (numpy linspace, truncated) plus the alerts
      const k = Math.min(TAPE_ROWS, n), tape = new Set(shown);
      for (let j = 0; j < k; j++) tape.add(cursor + (k === 1 ? 0 : j === k - 1 ? n - 1 : Math.trunc(j * ((n - 1) / (k - 1)))));
      yield ["tick", {
        sim_time: nxt, transactions: n,
        tape_rows: [...tape].sort((x, y) => x - y).map((i) => record(c, i, TAPE_COLS, { score: s[i] })),
        alerts: alertIdx.length, true_alerts: trueAlerts, frauds, amount,
        alert_rows: shown.map((i) => record(c, i, ALERT_COLS, { score: s[i] })),
      }];
      while (nxt >= day + DAY) { yield ["day", summary(day)]; day += DAY; }
      cursor = hi;
      sim = nxt;
    }
    yield ["end", { sim_time: sim }];
  }

  /** The part of EventSource that the tape screen uses, fed by replayEvents. */
  function openReplay(p) {
    const handlers = {};
    const src = {
      closed: false, onerror: null,
      addEventListener(event, fn) { handlers[event] = fn; },
      close() { this.closed = true; },
    };
    (async () => {
      try {
        let first = true;
        for await (const [event, data] of replayEvents(p)) {
          // One tick per TICK_SECONDS, like the server
          if (event === "tick" && !first) await new Promise((r) => setTimeout(r, TICK_SECONDS * 1000));
          if (event === "tick") first = false;
          if (src.closed) return;
          handlers[event]?.({ data: JSON.stringify(data) });
        }
      } catch (e) {
        if (!src.closed) src.onerror?.(e);
      }
    })();
    return src;
  }

  return { handle, openReplay, replayEvents };
}

/** The whole backend: /api/splits, and every other request routed by its `split`
 *  parameter (default: the API's default split) to that split's backend. */
export function createBackend(load = fetchFile) {
  let splitsInfo = null;
  const backends = new Map();
  const splits = () => (splitsInfo ??= load("splits.json")
    .then((b) => JSON.parse(new TextDecoder().decode(b)))
    .catch((e) => { splitsInfo = null; throw e; }));
  async function forSplit(params) {
    const { split, ...rest } = params ?? {};
    const info = await splits();
    const name = split ?? info.default;
    if (!info.splits.some((s) => s.split === name)) throw new ApiError(404, `split '${name}' not built`);
    if (!backends.has(name)) backends.set(name, createSplitBackend((file) => load(`${name}/${file}`)));
    return [backends.get(name), rest];
  }
  return {
    async handle(path, params = {}) {
      if (path.replace(/^\/api\//, "") === "splits") return splits();
      const [b, rest] = await forSplit(params);
      return b.handle(path, rest);
    },
    openReplay(params) {
      // EventSource-like object returned synchronously, as by the split backend
      let inner = null, closed = false;
      const handlers = {};
      const src = {
        onerror: null,
        addEventListener(event, fn) { handlers[event] = fn; inner?.addEventListener(event, fn); },
        close() { closed = true; inner?.close(); },
      };
      forSplit(params).then(([b, rest]) => {
        if (closed) return;
        inner = b.openReplay(rest);
        inner.onerror = (e) => src.onerror?.(e);
        for (const [event, fn] of Object.entries(handlers)) inner.addEventListener(event, fn);
      }).catch((e) => src.onerror?.(e));
      return src;
    },
    async *replayEvents(params) {
      const [b, rest] = await forSplit(params);
      yield* b.replayEvents(rest);
    },
  };
}
