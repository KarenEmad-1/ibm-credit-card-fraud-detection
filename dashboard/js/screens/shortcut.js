import { html, useState, useEffect, useApp, api, Pane, Tween, useD3, d3, tip, fmtPct, fmtInt, cssVar, modelColor, shortModel } from "../lib.js";

export function Shortcut() {
  const { split, meta, setStatus, keys } = useApp();
  const [geo, setGeo] = useState(true);
  const [abl, setAbl] = useState(null);
  const [seg, setSeg] = useState(null);
  const [curves, setCurves] = useState(null);
  const cb = meta.models.find((m) => m.key.startsWith("CatBoost"))?.key;

  useEffect(() => {
    api("/api/ablations").then(setAbl);
    if (cb) api(`/api/segments/${encodeURIComponent(cb)}`, {}, split).then((r) => setSeg(r.rows.filter((x) => x.segment === "Merchant location")));
    api("/api/curves", {}, split).then(setCurves);
  }, []);
  useEffect(() => {
    keys.current = (e) => { if (e.key.toLowerCase() === "g") { setGeo((g) => !g); return true; } return false; };
    return () => { keys.current = null; };
  }, []);
  useEffect(() => { setStatus(`GEOGRAPHY ${geo ? "ON" : "OFF"} · press G`); }, [geo]);

  const pr = (fam, v) => abl?.log.find((r) => r.model === fam && r.variant === v)?.pr_auc
    ?? meta.models.find((m) => m.key === `${fam} ${v}`)?.metrics.pr_auc;
  const a = pr("CatBoost", "A"), b = pr("CatBoost", "B"), c = pr("CatBoost", "C");
  const intl = seg?.find((x) => x.level === "International"), dom = seg?.find((x) => x.level === "Domestic in-store");
  const fams = ["LogReg", "RandomForest", "CatBoost", "FT-Transformer"];

  return html`<div class="stack">
    <div class="reveal">
      <${Pane} title="THE FINDING">
        <div class="headline">THE MODEL THAT<br /><em>READ THE MAP</em></div>
        <div class="prose"><p>The best model scores well. Then we hide one thing from it: <b>where the merchant is</b> (is_international and zip_freq, which is 0 for every foreign merchant).</p></div>
        <div style="display:flex;gap:28px;align-items:end;flex-wrap:wrap;margin:14px 0 6px">
          <div>
            <div class="muted" style="font-size:11px;letter-spacing:.14em">CATBOOST · VALIDATION PR-AUC</div>
            <div class=${`giant ${geo ? "" : "down"}`}><${Tween} value=${geo ? a : b} format=${(x) => (x == null ? "—" : x.toFixed(3))} ms=${1100} /></div>
          </div>
          <button class=${`toggle ${geo ? "on" : ""}`} onClick=${() => setGeo((g) => !g)} aria-pressed=${geo}>
            <kbd>G</kbd>GEOGRAPHY <span class="sw"></span> ${geo ? "ON" : "OFF"}</button>
        </div>
        <div class="prose"><p>${geo
          ? html`With geography the model looks excellent. Give it the raw merchant state as well and it climbs to <b>${c ? c.toFixed(3) : "—"}</b>.`
          : html`Without geography, <b>${a && b ? fmtPct(1 - b / a, 0) : "—"}</b> of the score is gone. That's how much of the "skill" was spotting a foreign merchant.`}</p></div>
      </${Pane}>

      <${Pane} title="FRAUD RATE BY MERCHANT LOCATION" right="log scale · 2018">
        ${seg ? html`<${RateBars} rows=${seg} />` : html`<div class="chart-box muted blink">loading…</div>`}
        <div class="facts">
          <div class="fact"><div class="n">${intl ? fmtPct(intl.share_of_all_fraud, 0) : "—"}</div><div class="t">of all fraud is international in-store</div></div>
          <div class="fact"><div class="n">${intl ? fmtPct(intl.fraud_rate, 1) : "—"}</div><div class="t">fraud rate at foreign merchants</div></div>
          <div class="fact"><div class="n">${intl && dom ? `${fmtInt(intl.fraud_rate / dom.fraud_rate)}×` : "—"}</div><div class="t">riskier than a domestic store</div></div>
        </div>
      </${Pane}>
    </div>

    <div class="grid g-2">
      <${Pane} title="EVERY MODEL DROPS" right="validation PR-AUC · variant A (with geography) → B (without)">
        <table class="tbl"><thead><tr><th>MODEL</th><th class="num">WITH</th><th></th><th class="num">WITHOUT</th><th class="num">LOST</th></tr></thead>
          <tbody>${fams.map((f) => {
            const va = pr(f, "A"), vb = pr(f, "B");
            return html`<tr><td><span style=${`color:${modelColor(f + " A")}`}>■</span> ${shortModel(f + " A").replace("-A", "")}</td>
              <td class="num">${va != null ? va.toFixed(3) : "—"}</td>
              <td><${Drop} a=${va} b=${vb} /></td>
              <td class="num ${vb != null ? "bad" : "muted"}">${vb != null ? vb.toFixed(3) : "running…"}</td>
              <td class="num">${va != null && vb != null ? fmtPct(1 - vb / va, 0) : "—"}</td></tr>`;
          })}</tbody></table>
        ${abl && !abl.complete ? html`<div class="muted" style="font-size:11px;margin-top:6px">notebook 07 still running: rows fill in as it logs results</div>` : null}
      </${Pane}>
      <${Pane} title="WHAT IT MEANS">
        <div class="prose">
          <p><b>The shortcut is real in the data and probably fake in the world.</b> In this synthetic generator, foreign in-store transactions are almost a fraud label. A real issuer's data wouldn't hand over the answer like that.</p>
          <p>So the project reports both numbers. <b>Variant B (no geography) is the headline</b>, and A shows what the shortcut is worth. Spotting this is the difference between a model that scores well and one that works.</p>
          <p class="muted">See [3] CASES: most stolen cards are caught the moment they're used abroad.</p>
        </div>
      </${Pane}>
    </div>
  </div>`;
}

function Drop({ a, b }) {
  if (a == null) return null;
  return html`<span style="display:inline-block;position:relative;width:120px;height:8px;background:var(--bg);border:1px solid var(--line);vertical-align:middle">
    <span style=${`position:absolute;left:0;top:0;bottom:0;width:${a * 100}%;background:var(--phos-2);opacity:.35`}></span>
    ${b != null ? html`<span style=${`position:absolute;left:0;top:0;bottom:0;width:${b * 100}%;background:var(--bad)`}></span>` : null}
  </span>`;
}

function RateBars({ rows }) {
  const ref = useD3((el, w, h) => {
    const svg = d3.select(el).append("svg").attr("class", "chart").attr("width", w).attr("height", h);
    const order = ["International", "Online", "Domestic in-store"];
    const data = order.map((l) => rows.find((r) => r.level === l)).filter(Boolean);
    const m = { t: 8, r: 70, b: 22, l: 130 };
    const x = d3.scaleLog().domain([1e-5, 1]).range([m.l, w - m.r]);
    const y = d3.scaleBand().domain(data.map((d) => d.level)).range([m.t, h - m.b]).padding(0.35);
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).tickValues([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]).tickSize(-(h - m.t - m.b)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).tickValues([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]).tickFormat((v) => fmtPct(v, v < 1e-3 ? 3 : 0)).tickSize(0)).select(".domain").remove();
    svg.selectAll(".lb").data(data).join("text").attr("x", m.l - 8).attr("y", (d) => y(d.level) + y.bandwidth() / 2 + 4).attr("text-anchor", "end")
      .style("fill", cssVar("--text-2")).style("font-size", "12px").text((d) => d.level.toUpperCase());
    svg.selectAll(".b").data(data).join("rect").attr("x", m.l).attr("y", (d) => y(d.level)).attr("height", y.bandwidth())
      .attr("width", (d) => Math.max(2, x(Math.max(d.fraud_rate, 1e-5)) - m.l)).attr("rx", 2)
      .attr("fill", (d) => (d.level === "International" ? cssVar("--bad") : cssVar("--phos-2")))
      .on("mousemove", (ev, d) => tip.show(ev, `<b>${d.level}</b><br>${fmtInt(d.transactions)} txns · ${fmtInt(d.frauds)} frauds<br>fraud rate ${fmtPct(d.fraud_rate, 3)}<br>${fmtPct(d.share_of_all_fraud, 1)} of all fraud`))
      .on("mouseleave", () => tip.hide());
    svg.selectAll(".v").data(data).join("text").attr("x", (d) => x(Math.max(d.fraud_rate, 1e-5)) + 6).attr("y", (d) => y(d.level) + y.bandwidth() / 2 + 4)
      .style("fill", cssVar("--text")).style("font-size", "12px").text((d) => fmtPct(d.fraud_rate, d.fraud_rate < 1e-3 ? 4 : 2));
  }, [rows]);
  return html`<div class="chart-box" style="height:170px" ref=${ref}></div>`;
}
