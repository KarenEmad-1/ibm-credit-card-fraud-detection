import { html, useState, useEffect, useRef, useApp, api, Pane, useD3, d3, tip, fmtPct, fmtInt, fmtK, fmt3, cssVar, modelColor,
  shortModel, variantOf } from "../lib.js";

export function Lab() {
  const { split, meta, model, modelMeta, setStatus } = useApp();
  const [curves, setCurves] = useState(null);
  const [t, setT] = useState(modelMeta.threshold);
  const [r, setR] = useState(null);
  const [seg, setSeg] = useState(null);
  const req = useRef(0);

  useEffect(() => { api("/api/curves", {}, split).then(setCurves); }, []);
  useEffect(() => { setT(modelMeta.threshold); }, [model]);
  useEffect(() => {
    const id = ++req.current;
    api(`/api/threshold/${encodeURIComponent(model)}`, { t }, split).then((x) => { if (id === req.current) setR(x); });
  }, [t, model]);
  useEffect(() => { setSeg(null); api(`/api/segments/${encodeURIComponent(model)}`, {}, split).then(setSeg); }, [model]);
  useEffect(() => { setStatus(`${meta.split === "test" ? "TEST" : "VAL"}-${meta.year} · ${fmtInt(meta.n_transactions)} TXNS · ${fmtInt(meta.n_frauds)} FRAUD`); }, []);

  const best = (f) => Math.max(...meta.models.map((m) => m.metrics[f]));
  return html`<div class="stack">
    <div class="grid g-lab">
      <${Pane} title="PRECISION · RECALL" right=${curves ? `random = ${fmtPct(curves.base_rate, 2)}` : ""}>
        <div class="legend">${meta.models.map((m) => html`<span><i style=${`background:${modelColor(m.key)}`}></i>${shortModel(m.key)} ${m.metrics.pr_auc.toFixed(3)}</span>`)}</div>
        ${curves ? html`<${PRChart} curves=${curves} models=${meta.models} />` : html`<div class="chart-box tall muted blink">loading…</div>`}
      </${Pane}>
      <${Pane} cls="flush" title="MODEL BOARD" right="thresholds chosen on validation (max F1)">
        <table class="tbl">
          <thead><tr><th>MODEL</th><th class="num">PR-AUC</th><th class="num">R@P≥.5</th><th class="num">P@TOP.1%</th><th class="num">P@THR</th><th class="num">R@THR</th><th class="num">ALERTS</th></tr></thead>
          <tbody>${meta.models.map((m) => {
            const c = (f, fn) => html`<td class=${`num ${m.metrics[f] === best(f) ? "phos" : ""}`}>${fn(m.metrics[f])}</td>`;
            return html`<tr class=${m.key === model ? "sel" : ""}><td><span style=${`color:${modelColor(m.key)}`}>■</span> ${shortModel(m.key)}</td>
              ${c("pr_auc", fmt3)}${c("recall@precision0.5", fmtPct)}${c("precision@top0.1%", fmtPct)}
              ${c("precision@threshold", fmtPct)}${c("recall@threshold", fmtPct)}<td class="num">${fmtInt(m.metrics["alerts@threshold"])}</td></tr>`;
          })}</tbody>
        </table>
        <div class="prose" style="padding:10px">
          <p>PR-AUC is the headline because fraud is 0.14% of traffic. At that rate ROC-AUC is above 0.98 for every model and can't tell them apart. FT-Transformer's ${fmt3(meta.models.find((m) => m.key.startsWith("FT"))?.metrics.pr_auc)} is the 3-seed average after stabilising, with no picking of the best epoch on validation.</p>
        </div>
      </${Pane}>
    </div>

    <div class="grid g-lab">
      <${Pane} title=${`THRESHOLD SIMULATOR · ${shortModel(model)}`} right=${html`<button class="btn" onClick=${() => setT(modelMeta.threshold)}>RESET TO VAL ${modelMeta.threshold.toFixed(3)}</button>`}>
        <div class="ctl" style="margin-bottom:10px">THR
          <input type="range" min="0" max="1000" value=${Math.round(t * 1000)} style="flex:1"
            onInput=${(e) => { const v = +e.target.value; setT(v === Math.round(modelMeta.threshold * 1000) ? modelMeta.threshold : v / 1000); }} />
          <b>${t.toFixed(3)}</b></div>
        ${r ? html`<div class="kpis">
          ${k("ALERTS/DAY", r.alerts_per_day.toFixed(1), `${fmtInt(r.alerts)} in the year`)}
          ${k("PRECISION", fmtPct(r.precision), "alerts that are fraud")}
          ${k("RECALL", fmtPct(r.recall), `${fmtInt(r.fn)} frauds missed`)}
          ${k("FALSE/CATCH", (r.fp / Math.max(r.tp, 1)).toFixed(2), `${fmtInt(r.fp)} false alerts`, "bad")}
          ${k("$ CAUGHT", fmtK(r.fraud_amount_caught), `${fmtPct(r.fraud_amount_caught / r.fraud_amount_total)} of fraud $`, "ok")}
        </div>` : null}
        ${curves ? html`<${ThrChart} c=${curves.curves[model]} t=${t} vt=${modelMeta.threshold} />` : null}
      </${Pane}>
      <${Pane} cls="flush" title=${`SEGMENTS · ${shortModel(model)}`} right="at the validation threshold">
        <div class="scroll" style="max-height:470px">
          <table class="tbl">
            <thead><tr><th>SEGMENT</th><th class="num">TXNS</th><th class="num">FRAUD</th><th class="num">RATE</th><th class="num">PR-AUC</th><th class="num">PREC</th><th class="num">RECALL</th></tr></thead>
            <tbody>${seg ? seg.rows.map((x, i) => html`
              ${i === 0 || seg.rows[i - 1].segment !== x.segment ? html`<tr><td colspan="7" class="phos" style="padding-top:8px">${x.segment.toUpperCase()}</td></tr>` : null}
              <tr><td>  ${x.level}</td><td class="num muted">${fmtInt(x.transactions)}</td><td class="num">${fmtInt(x.frauds)}</td>
                <td class="num">${fmtPct(x.fraud_rate, 3)}</td><td class="num">${fmt3(x.pr_auc)}</td>
                <td class="num">${fmtPct(x.precision)}</td><td class=${`num ${x.recall != null && x.recall < 0.5 ? "bad" : ""}`}>${fmtPct(x.recall)}</td></tr>`)
              : html`<tr><td colspan="7" class="muted blink">computing segments…</td></tr>`}</tbody>
          </table>
        </div>
      </${Pane}>
    </div>
  </div>`;
}

const k = (l, n, s, cls = "") => html`<div class="kpi"><div class="l">${l}</div><div class=${`n ${cls}`}>${n}</div><div class="s">${s}</div></div>`;

function PRChart({ curves, models }) {
  const ref = useD3((el, w, h) => {
    const svg = d3.select(el).append("svg").attr("class", "chart").attr("width", w).attr("height", h);
    const m = { t: 8, r: 10, b: 30, l: 40 };
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([h - m.b, m.t]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(5).tickSize(-(h - m.t - m.b)).tickFormat(""));
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(w - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat((v) => fmtPct(v, 0)).tickSize(0)).select(".domain").remove();
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat((v) => fmtPct(v, 0)).tickSize(0)).select(".domain").remove();
    svg.append("text").attr("x", w - m.r).attr("y", h - 4).attr("text-anchor", "end").text("RECALL →");
    svg.append("text").attr("x", m.l + 4).attr("y", m.t + 10).text("↑ PRECISION");
    svg.append("line").attr("x1", x(0)).attr("x2", x(1)).attr("y1", y(curves.base_rate)).attr("y2", y(curves.base_rate)).attr("stroke", cssVar("--dim"));
    const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
    const series = models.map((mm) => {
      const c = curves.curves[mm.key];
      return { key: mm.key, pts: c.recall.map((rv, i) => [rv, c.precision[i], c.threshold[i]]) };
    });
    series.forEach((s) => svg.append("path").datum(s.pts).attr("fill", "none").attr("stroke", modelColor(s.key)).attr("stroke-width", 2)
      .attr("stroke-dasharray", variantOf(s.key) === "A" ? null : "6 4").attr("d", line));
    const dots = svg.append("g");
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", w - m.l - m.r).attr("height", h - m.t - m.b).attr("fill", "transparent")
      .on("mousemove", (ev) => {
        const rec = x.invert(d3.pointer(ev)[0]);
        const near = series.map((s) => ({ key: s.key, p: s.pts.reduce((a, b) => (Math.abs(b[0] - rec) < Math.abs(a[0] - rec) ? b : a)) }));
        dots.selectAll("circle").data(near).join("circle").attr("r", 4).attr("cx", (d) => x(d.p[0])).attr("cy", (d) => y(d.p[1]))
          .attr("fill", (d) => modelColor(d.key)).attr("stroke", cssVar("--bg-2")).attr("stroke-width", 2);
        tip.show(ev, `<b>recall ≈ ${fmtPct(rec, 0)}</b><br>` + near.map((d) => `<span style="color:${modelColor(d.key)}">■</span> ${shortModel(d.key)}: precision ${fmtPct(d.p[1])} @ thr ${d.p[2].toFixed(3)}`).join("<br>"));
      })
      .on("mouseleave", () => { tip.hide(); dots.selectAll("*").remove(); });
  }, [curves]);
  return html`<div class="chart-box tall" ref=${ref}></div>`;
}

function ThrChart({ c, t, vt }) {
  const ref = useD3((el, w, h) => {
    const svg = d3.select(el).append("svg").attr("class", "chart").attr("width", w).attr("height", h);
    const m = { t: 18, r: 10, b: 26, l: 40 };
    const x = d3.scaleLinear().domain([0, 1]).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([h - m.b, m.t]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(-(w - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(5).tickSize(0)).select(".domain").remove();
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat((v) => fmtPct(v, 0)).tickSize(0)).select(".domain").remove();
    const pts = c.threshold.map((tt, i) => ({ t: tt, p: c.precision[i], r: c.recall[i] })).sort((a, b) => a.t - b.t);
    const P = cssVar("--m1"), R = cssVar("--m2");
    svg.append("path").datum(pts).attr("fill", "none").attr("stroke", P).attr("stroke-width", 2).attr("d", d3.line().x((d) => x(d.t)).y((d) => y(d.p)));
    svg.append("path").datum(pts).attr("fill", "none").attr("stroke", R).attr("stroke-width", 2).attr("d", d3.line().x((d) => x(d.t)).y((d) => y(d.r)));
    svg.append("text").attr("x", m.l).attr("y", 11).style("fill", P).text("■ PRECISION");
    svg.append("text").attr("x", m.l + 96).attr("y", 11).style("fill", R).text("■ RECALL");
    svg.append("line").attr("x1", x(vt)).attr("x2", x(vt)).attr("y1", m.t).attr("y2", h - m.b).attr("stroke", cssVar("--dim")).attr("stroke-dasharray", "3 3");
    svg.append("line").attr("x1", x(t)).attr("x2", x(t)).attr("y1", m.t).attr("y2", h - m.b).attr("stroke", cssVar("--phos")).attr("stroke-width", 1.5);
  }, [c, t, vt]);
  return html`<div class="chart-box" style="height:200px;margin-top:10px" ref=${ref}></div>`;
}
