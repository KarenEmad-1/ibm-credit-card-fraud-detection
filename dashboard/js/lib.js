// Shared imports, API client, formatters and small UI pieces for FRAUD-OPS.
// No build step: Preact (via htm's standalone bundle) and D3 load as ES modules.
// "htm/preact/standalone" and "d3" are resolved by the import map in index.html: CDN
// URLs when served from the repo, local copies in the Docker image (see build.mjs).
export {
  html, render, useState, useEffect, useRef, useMemo, useCallback, useContext, createContext,
} from "htm/preact/standalone";
import { html, useEffect, useRef, createContext, useContext } from "htm/preact/standalone";
export * as d3 from "d3";

// ------------------------------------------------------------------ app context
export const Ctx = createContext(null);
export const useApp = () => useContext(Ctx);

// ------------------------------------------------------------------ api
export async function api(path, params = {}, split = null) {
  const q = new URLSearchParams(Object.fromEntries(Object.entries({ ...params, split }).filter(([, v]) => v != null)));
  const r = await fetch(`${path}?${q}`);
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

// ------------------------------------------------------------------ format
export const fmtInt = (n) => (n == null ? "—" : Math.round(n).toLocaleString("en-US"));
export const fmtPct = (x, d = 1) => (x == null || isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`);
export const fmt3 = (x) => (x == null ? "—" : x.toFixed(3));
export const fmtMoney = (x, d = 2) => (x == null ? "—" : `$${Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })}`);
export const fmtK = (x) => (x == null ? "—" : Math.abs(x) >= 1e6 ? `$${(x / 1e6).toFixed(2)}M` : Math.abs(x) >= 1e3 ? `$${(x / 1e3).toFixed(1)}k` : `$${x.toFixed(0)}`);
export const tsDate = (s) => new Date(s * 1000).toISOString().slice(0, 10);
export const tsTime = (s) => new Date(s * 1000).toISOString().slice(11, 16);
export const tsFull = (s) => new Date(s * 1000).toISOString().replace("T", " ").slice(0, 16);
export const place = (r) => (r.city === "ONLINE" ? "ONLINE" : `${r.city}${r.state ? ", " + r.state : ""}`);
export const upper = (s) => String(s ?? "").toUpperCase();
export function ago(sec) {
  const m = Math.round(sec / 60);
  if (m < 60) return `${m}m`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h`;
  return `${Math.round(h / 24)}d`;
}

// Merchant category codes that show up most in this data.
const MCC = {
  5411: "grocery", 5812: "restaurant", 5814: "fast food", 5813: "bar", 5541: "gas station", 5912: "pharmacy",
  5311: "department store", 5310: "discount store", 5300: "wholesale club", 5211: "building supplies",
  4829: "money transfer", 4121: "taxi / rideshare", 4784: "tolls", 5815: "digital goods", 5499: "food store",
  7538: "auto repair", 5942: "bookstore", 5651: "clothing", 5045: "computers", 4900: "utilities",
  7832: "cinema", 5732: "electronics", 5719: "home furnishing", 5661: "shoes", 4814: "telecom",
  7011: "hotel", 3000: "airline", 3001: "airline", 3005: "airline", 3006: "airline", 3007: "airline",
  3008: "airline", 3009: "airline", 4511: "airline", 8021: "dentist", 8011: "doctor", 8099: "health",
  8043: "optician", 8062: "hospital", 7349: "cleaning", 7801: "gambling", 5193: "florist",
  5970: "art supplies", 5733: "music store", 5094: "jewellery", 5921: "liquor store", 7230: "salon",
  5192: "newsstand", 4214: "freight", 7393: "security", 7531: "body shop", 7549: "towing", 3504: "hotel",
  3509: "hotel", 3640: "hotel", 3684: "hotel", 3132: "airline", 3058: "airline", 3075: "airline",
  3174: "airline", 3256: "airline", 3260: "airline", 3359: "car rental", 3387: "car rental",
  3389: "car rental", 3390: "car rental", 3393: "car rental", 3395: "car rental", 3405: "car rental",
  5712: "furniture", 5251: "hardware", 5261: "garden", 6300: "insurance", 7996: "amusement park",
  7995: "betting", 9402: "postal", 5655: "sportswear", 5947: "gift shop", 5977: "cosmetics", 5932: "antiques",
};
export const mccName = (m) => MCC[m] || "merchant";

// ------------------------------------------------------------------ colors
// Model identity uses the categorical slots in fixed order; never by rank.
const FAMILY_VAR = { "LogReg": "--m1", "RandomForest": "--m2", "CatBoost": "--m3", "FT-Transformer": "--m4" };
export const familyOf = (k) => k.split(" ").slice(0, -1).join(" ");
export const variantOf = (k) => k.split(" ").slice(-1)[0];
export const cssVar = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
export const modelColor = (k) => cssVar(FAMILY_VAR[familyOf(k)] || "--text");
export const shortModel = (k) => upper(k).replace("RANDOMFOREST", "RF").replace("FT-TRANSFORMER", "FT-TRANS").replace(" ", "-");

// ------------------------------------------------------------------ ui pieces
export function Pane({ title, right, children, cls = "", style }) {
  return html`<section class="pane ${cls}" style=${style}>
    <header class="pane-h"><span class="pane-t">${title}</span>${right ? html`<span class="pane-r">${right}</span>` : null}</header>
    <div class="pane-b">${children}</div>
  </section>`;
}

export const Key = ({ k, children, onClick, active }) =>
  html`<button type="button" class="key ${active ? "on" : ""}" onClick=${onClick}><kbd>${k}</kbd>${children}</button>`;

/** ASCII-ish score bar: ▕████░░░░▏ with the threshold as a tick. */
export function ScoreBar({ score, thr, width = 12 }) {
  const n = Math.round(score * width);
  const t = thr == null ? -1 : Math.min(width - 1, Math.floor(thr * width));
  const cells = Array.from({ length: width }, (_, i) => (i < n ? "█" : i === t ? "┊" : "░"));
  return html`<span class="sbar ${thr != null && score >= thr ? "hot" : ""}">${cells.join("")}</span>`;
}

export const Verdict = ({ fraud, pending }) =>
  pending ? html`<span class="v pend">◷ PENDING</span>`
    : fraud ? html`<span class="v bad">▲ FRAUD</span>` : html`<span class="v ok">✓ CLEAR</span>`;

/** Number that counts toward its target when it changes. */
export function Tween({ value, format = (x) => x, ms = 900 }) {
  const ref = useRef(null);
  const last = useRef(value);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const from = last.current ?? value, to = value;
    last.current = value;
    if (from === to) { el.textContent = format(to); return; }
    if (from == null || to == null || matchMedia("(prefers-reduced-motion: reduce)").matches) { el.textContent = format(to); return; }
    const t0 = performance.now();
    let raf;
    const step = (t) => {
      const p = Math.min(1, (t - t0) / ms), e = 1 - (1 - p) ** 3;
      el.textContent = format(from + (to - from) * e);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  // the effect owns the text; rendering children too would make Preact and the tween both write it
  return html`<span ref=${ref}></span>`;
}

/** Run a D3 draw function against an <svg> sized to its container. */
export function useD3(draw, deps) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let cleanup;
    const run = () => {
      const { width, height } = el.getBoundingClientRect();
      el.innerHTML = "";
      cleanup?.();
      cleanup = draw(el, Math.max(width, 10), Math.max(height, 10));
    };
    run();
    const ro = new ResizeObserver(() => run());
    ro.observe(el);
    return () => { ro.disconnect(); cleanup?.(); };
  }, deps);
  return ref;
}

/** Floating tooltip shared by all charts. */
export const tip = {
  el: null,
  show(evt, content) {
    if (!this.el) { this.el = document.createElement("div"); this.el.className = "tip"; document.body.appendChild(this.el); }
    this.el.innerHTML = content;
    this.el.style.display = "block";
    const pad = 14, w = this.el.offsetWidth, h = this.el.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > innerHeight - 8) y = evt.clientY - h - pad;
    this.el.style.left = `${x}px`; this.el.style.top = `${y}px`;
  },
  hide() { if (this.el) this.el.style.display = "none"; },
};
