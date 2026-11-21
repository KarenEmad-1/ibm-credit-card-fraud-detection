import { html, render, useState, useEffect, useRef, useCallback, Ctx, api, fmtInt, upper, shortModel, modelColor, Key } from "./lib.js";
import { Tape } from "./screens/tape.js";
import { Hunt } from "./screens/hunt.js";
import { Cases } from "./screens/cases.js";
import { Triage } from "./screens/triage.js";
import { Shortcut } from "./screens/shortcut.js";
import { Lab } from "./screens/lab.js";

const SCREENS = [
  { id: "tape", key: "1", label: "TAPE", title: "Live tape", Comp: Tape,
    brief: "The year replayed in time order. Every row is a transaction scored by the model. ▲ rows go to the alert queue, and their labels arrive days later, as chargebacks do." },
  { id: "hunt", key: "2", label: "HUNT", title: "Beat the model", Comp: Hunt,
    brief: "You're the fraud analyst. Judge 10 transactions as fraud or legit, then see whether you beat the model." },
  { id: "cases", key: "3", label: "CASES", title: "Card attack files", Comp: Cases,
    brief: "Replay one stolen card: normal spending, then the fraud burst. How long did the model take to notice, and how much was spent first?" },
  { id: "triage", key: "4", label: "TRIAGE", title: "Alert triage", Comp: Triage,
    brief: "One shift, a limited number of investigations. Pick which alerts to work, then compare your dollars recovered with the model's ranking." },
  { id: "shortcut", key: "5", label: "SHORTCUT", title: "The shortcut", Comp: Shortcut,
    brief: "The best model scores 0.86. Hide merchant geography and see what's left. This is the project's main finding." },
  { id: "lab", key: "6", label: "LAB", title: "Model lab", Comp: Lab,
    brief: "The evidence behind the screens: PR curves, a threshold simulator and per-segment performance for all four models." },
];

const BOOT_LINES = (m) => [
  ["FRAUD-OPS v1.0 · card-fraud operations console", ""],
  [`mounting ledger ${m.split === "test" ? "TEST-2019" : "VAL-2018"}`, `${fmtInt(m.n_transactions)} txns`],
  ["labelled fraud", `${fmtInt(m.n_frauds)} (1 in ${fmtInt(m.n_transactions / m.n_frauds)})`],
  ...m.models.map((x) => [`model ${shortModel(x.key)}`, `PR-AUC ${x.metrics.pr_auc.toFixed(3)}`]),
  ["decision thresholds", "LOCKED · chosen on validation"],
  ["2019 test ledger", m.split === "test" ? "OPEN · scored once" : "SEALED until notebook 08"],
  ["", "READY"],
];

function Boot({ meta, onDone }) {
  const [n, setN] = useState(0);
  const lines = BOOT_LINES(meta);
  useEffect(() => {
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { onDone(); return; }
    const id = setInterval(() => setN((x) => x + 1), 170);
    const skip = () => onDone();
    window.addEventListener("keydown", skip, { once: true });
    window.addEventListener("pointerdown", skip, { once: true });
    return () => { clearInterval(id); window.removeEventListener("keydown", skip); window.removeEventListener("pointerdown", skip); };
  }, []);
  useEffect(() => { if (n > lines.length + 3) onDone(); }, [n]);
  return html`<div class="boot" role="status">
    <pre>${lines.slice(0, n).map(([a, b], i) => html`<div class="boot-l">
      <span>${a ? `> ${a}` : ""}</span><span class="dots">${a ? " ".padEnd(Math.max(2, 46 - a.length), ".") : ""}</span>
      <span class=${i === lines.length - 1 ? "ok blink" : "ok"}>${b}</span></div>`)}</pre>
    <div class="boot-skip">press any key</div>
  </div>`;
}

function Help({ onClose }) {
  return html`<div class="overlay" onClick=${onClose}>
    <div class="help" onClick=${(e) => e.stopPropagation()}>
      <div class="pane-h"><span class="pane-t">HELP · ABOUT THIS CONSOLE</span><span class="pane-r">[ESC] close</span></div>
      <div class="help-b">
        <p><b>What this is.</b> Fraud detection on 24.4M synthetic card transactions (IBM, 1991–2020). The models are logistic regression, random forest, CatBoost and an FT-Transformer, trained on 1991–2017 and validated on 2018. 2019 is a test year scored exactly once.</p>
        <p><b>Rules the project follows.</b> The split is chronological, per-card features use only past transactions, no feature uses fraud labels, thresholds are chosen on validation, and the test year is sealed until the end.</p>
        <p><b>Main finding.</b> Most of the score comes from merchant geography. In this synthetic data, foreign in-store transactions are 94% of fraud. See [5] SHORTCUT.</p>
        <table class="keys">
          <tr><td><kbd>1</kbd>–<kbd>6</kbd></td><td>switch screen</td><td><kbd>M</kbd></td><td>next model</td></tr>
          <tr><td><kbd>SPACE</kbd></td><td>play / pause tape</td><td><kbd>+</kbd> <kbd>−</kbd></td><td>tape speed</td></tr>
          <tr><td><kbd>F</kbd> <kbd>L</kbd> <kbd>N</kbd></td><td>hunt: fraud / legit / next</td><td><kbd>T</kbd></td><td>amber / green phosphor</td></tr>
          <tr><td><kbd>?</kbd></td><td>this help</td><td><kbd>ESC</kbd></td><td>close</td></tr>
        </table>
      </div>
    </div>
  </div>`;
}

function App() {
  const [splits, setSplits] = useState(null);
  const [split, setSplit] = useState(null);
  const [meta, setMeta] = useState(null);
  const [model, setModel] = useState(null);
  const [screen, setScreen] = useState(() => { try { return sessionStorage.getItem("fo-screen") || "tape"; } catch { return "tape"; } });
  const [booted, setBooted] = useState(() => { try { return sessionStorage.getItem("fo-booted") === "1"; } catch { return false; } });
  const [help, setHelp] = useState(false);
  const [status, setStatus] = useState("");
  const [err, setErr] = useState(null);
  const keys = useRef(null);

  useEffect(() => { api("/api/splits").then((s) => { setSplits(s); setSplit(s.default); }).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    if (!split) return;
    Promise.all([api("/api/models", {}, split), api("/api/drift", {}, split)]).then(([m, drift]) => {
      setMeta({ ...m, split, drift });
      setModel((cur) => (m.models.some((x) => x.key === cur) ? cur : (m.models.find((x) => x.key.startsWith("CatBoost")) || m.models[0]).key));
    }).catch((e) => setErr(e.message));
  }, [split]);
  useEffect(() => { try { sessionStorage.setItem("fo-screen", screen); } catch { /* ignore */ } setStatus(""); }, [screen]);

  const cycleModel = useCallback(() => {
    if (!meta) return;
    const ks = meta.models.map((m) => m.key);
    setModel((cur) => ks[(ks.indexOf(cur) + 1) % ks.length]);
  }, [meta]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest("input, select, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key;
      if (help) { if (k === "Escape" || k === "?") setHelp(false); return; }
      const s = SCREENS.find((x) => x.key === k);
      if (s) { setScreen(s.id); e.preventDefault(); return; }
      if (k === "?") { setHelp(true); return; }
      if (k === "m" || k === "M") { cycleModel(); return; }
      if (k === "t" || k === "T") { document.documentElement.dataset.phosphor = document.documentElement.dataset.phosphor === "green" ? "amber" : "green"; return; }
      if (keys.current && keys.current(e)) e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [help, cycleModel]);

  if (err) return html`<div class="fatal"><pre>FRAUD-OPS · cannot reach the API

${err}

start it with:  uvicorn api.main:app --port 8000
build data with: python scripts/build_dashboard_data.py --split val</pre></div>`;
  if (!meta || !model) return html`<div class="fatal"><pre class="blink">connecting…</pre></div>`;
  if (!booted) return html`<${Boot} meta=${meta} onDone=${() => { setBooted(true); try { sessionStorage.setItem("fo-booted", "1"); } catch { /* ignore */ } }} />`;

  const cur = SCREENS.find((s) => s.id === screen) || SCREENS[0];
  const mm = meta.models.find((x) => x.key === model);
  const ctx = { meta, split, model, modelMeta: mm, setModel, setStatus, keys, go: setScreen };
  const Comp = cur.Comp;

  return html`<${Ctx.Provider} value=${ctx}>
    <div class="shell">
      <header class="top">
        <div class="logo">FRAUD<span>-</span>OPS</div>
        <div class="crumbs">
          <label class="crumb">LEDGER
            <select value=${split} onChange=${(e) => setSplit(e.target.value)} aria-label="Data split">
              ${splits.splits.map((s) => html`<option value=${s.split}>${s.split === "test" ? "TEST" : "VAL"}-${s.year}</option>`)}
            </select></label>
          <span class="sep">▸</span>
          <label class="crumb">MODEL
            <select value=${model} onChange=${(e) => setModel(e.target.value)} aria-label="Model">
              ${meta.models.map((m) => html`<option value=${m.key}>${shortModel(m.key)}</option>`)}
            </select>
            <span class="dot" style=${`background:${modelColor(model)}`}></span></label>
          <span class="sep">▸</span>
          <span class="crumb">THR <b>${mm.threshold.toFixed(3)}</b></span>
          <span class="sep hide-s">▸</span>
          <span class="crumb hide-s">PR-AUC <b>${mm.metrics.pr_auc.toFixed(3)}</b></span>
        </div>
        <div class="status">${status}</div>
      </header>

      <div class="brief"><span class="brief-t">${cur.key} · ${upper(cur.title)}</span><span class="brief-x">${cur.brief}</span></div>

      <main class="screen">${html`<${Comp} key=${`${screen}-${split}`} />`}</main>

      <nav class="fkeys" aria-label="Screens">
        ${SCREENS.map((s) => html`<${Key} k=${s.key} active=${s.id === screen} onClick=${() => setScreen(s.id)}>${s.label}</${Key}>`)}
        <span class="grow"></span>
        <${Key} k="M" onClick=${cycleModel}>MODEL</${Key}>
        <${Key} k="?" onClick=${() => setHelp(true)}>HELP</${Key}>
      </nav>
      ${help ? html`<${Help} onClose=${() => setHelp(false)} />` : null}
    </div>
  </${Ctx.Provider}>`;
}

render(html`<${App} />`, document.getElementById("app"));
