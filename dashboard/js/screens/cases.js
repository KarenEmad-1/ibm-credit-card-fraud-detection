import { html, useState, useEffect, useMemo, useApp, api, Pane, useD3, d3, tip, fmtMoney, fmtInt, fmtK, tsFull, tsDate,
  place, upper, mccName, cssVar, shortModel, period } from "../lib.js";

export function Cases() {
  const { split, model, setStatus, meta } = useApp();
  const [cards, setCards] = useState(null);
  const [sel, setSel] = useState(null);
  const [tl, setTl] = useState(null);
  const [domain, setDomain] = useState(null);
  const [focus, setFocus] = useState(null);
  const [why, setWhy] = useState(null);

  useEffect(() => {
    api("/api/cards/attacks", { model, limit: 40 }, split).then((r) => { setCards(r.cards); setSel((s) => s || r.cards[0]); });
  }, [model]);
  useEffect(() => {
    if (!sel) return;
    setTl(null); setFocus(null); setWhy(null);
    api(`/api/cards/${sel.user}/${sel.card}`, { model }, split).then((r) => { setTl(r); setDomain(r.attack_window); });
  }, [sel, model]);
  useEffect(() => {
    if (!focus) return;
    setWhy(null);
    const ex = meta.explainable[0];
    if (ex) api(`/api/explain/${encodeURIComponent(ex)}/${focus.txn_id}`, {}, split).then(setWhy).catch(() => setWhy(null));
  }, [focus]);

  const story = useMemo(() => tl && detection(tl), [tl]);
  useEffect(() => { if (sel) setStatus(`CASE u${sel.user}·c${sel.card} · ${sel.frauds} frauds`); }, [sel]);

  return html`<div class="grid g-cases">
    <${Pane} cls="flush" title="COMPROMISED CARDS" right=${`top 40 · ${period(meta)}`}>
      <div class="scroll" style="max-height: 640px">
        <table class="tbl click">
          <thead><tr><th>CARD</th><th class="num">FRAUD</th><th class="num">CAUGHT</th><th>WHERE</th></tr></thead>
          <tbody>${(cards || []).map((c) => html`<tr class=${sel && sel.user === c.user && sel.card === c.card ? "sel" : ""} onClick=${() => setSel(c)}>
            <td>u${c.user}·c${c.card}</td><td class="num">${c.frauds}</td>
            <td class="num ${c.caught / c.frauds >= 0.5 ? "ok" : "bad"}">${c.caught}/${c.frauds}</td>
            <td class="muted">${upper(c.city).slice(0, 12)}</td></tr>`)}</tbody>
        </table>
      </div>
    </${Pane}>

    <div class="stack">
      <${Pane} title=${sel ? `CASE FILE · u${sel.user}·c${sel.card}` : "CASE FILE"} right=${tl ? `${fmtInt(tl.rows.length)} transactions in ${period(meta)} · drag the bottom strip to zoom` : ""}>
        ${story ? html`<div class="case-sum">
          <span>FIRST FRAUD <b>${tsFull(story.firstFraud)}</b></span>
          <span>FIRST CATCH ${story.firstCatch ? html`<b>${tsFull(story.firstCatch)}</b>` : html`<b class="bad">NEVER</b>`}</span>
          <span>LET THROUGH BEFORE CATCH <b class="bad">${story.before} frauds · ${fmtMoney(story.beforeAmt)}</b></span>
          <span>CAUGHT <b>${story.caught}/${story.frauds}</b></span>
          <span>FALSE ALERTS <b>${story.falseAlerts}</b></span>
        </div>` : null}
        <div class="legend">
          <span><i style="background:var(--dim);border-radius:50%"></i>legit</span>
          <span><i style="background:var(--bad);clip-path:polygon(50% 0,100% 100%,0 100%)"></i>fraud (truth)</span>
          <span><i style="border:2px solid var(--phos);background:none;border-radius:50%"></i>flagged by ${shortModel(model)}</span>
          <span><i style="background:var(--phos)"></i>risk score</span>
        </div>
        ${tl ? html`<${Timeline} tl=${tl} domain=${domain} setDomain=${setDomain} onPick=${setFocus} focus=${focus} />` : html`<div class="chart-box tall muted blink">loading timeline…</div>`}
      </${Pane}>

      <${Pane} title="EVIDENCE" right=${focus ? `txn ${focus.txn_id}` : "click a point on the timeline"}>
        ${!focus ? html`<div class="muted">Pick any point to see the transaction and its SHAP explanation.</div>` : html`
          <div class="case-sum">
            <span>${tsFull(focus.ts)}</span><span><b>${fmtMoney(focus.amount)}</b></span><span>${upper(place(focus))}</span>
            <span>${mccName(focus.mcc)}</span><span>${upper(focus.channel)}</span>
            <span>score <b>${focus.score.toFixed(3)}</b></span>
            <span class=${focus.is_fraud ? "bad" : "ok"}>${focus.is_fraud ? "▲ FRAUD" : "✓ LEGIT"}</span>
          </div>
          ${why ? html`<div>${why.contributions.slice(0, 6).map((r) => html`<div class="reason">
            <span>${r.feature} = <b>${r.value ?? "missing"}</b></span>
            <span class=${r.shap > 0 ? "bad" : "ok"}>${r.shap > 0 ? "▲" : "▼"} ${Math.abs(r.shap).toFixed(2)}</span></div>`)}
            <div class="muted" style="font-size:11px">${shortModel(why.model)} SHAP, log-odds · ▲ toward fraud</div></div>` : html`<div class="muted blink">explaining…</div>`}`}
      </${Pane}>
    </div>
  </div>`;
}

function detection(tl) {
  const rows = tl.rows, thr = tl.threshold;
  const frauds = rows.filter((r) => r.is_fraud);
  const caughtRows = frauds.filter((r) => r.score >= thr);
  const firstCatch = caughtRows.length ? caughtRows[0].ts : null;
  const before = frauds.filter((r) => firstCatch == null || r.ts < firstCatch);
  return {
    frauds: frauds.length, caught: caughtRows.length, firstFraud: frauds[0]?.ts, firstCatch,
    before: before.length, beforeAmt: before.reduce((t, r) => t + Math.abs(r.amount), 0),
    falseAlerts: rows.filter((r) => !r.is_fraud && r.score >= thr).length,
  };
}

function Timeline({ tl, domain, setDomain, onPick, focus }) {
  const ref = useD3((el, w, h) => {
    const rows = tl.rows, thr = tl.threshold;
    const svg = d3.select(el).append("svg").attr("class", "chart").attr("width", w).attr("height", h);
    const m = { l: 52, r: 12 };
    const H1 = [8, h * 0.56], H2 = [h * 0.56 + 18, h * 0.82], H3 = [h * 0.82 + 22, h - 4];
    const full = d3.extent(rows, (r) => r.ts);
    const x = d3.scaleUtc().domain((domain || full).map((s) => new Date(s * 1000))).range([m.l, w - m.r]);
    const xc = d3.scaleUtc().domain(full.map((s) => new Date(s * 1000))).range([m.l, w - m.r]);
    const inView = rows.filter((r) => r.ts * 1000 >= x.domain()[0] && r.ts * 1000 <= x.domain()[1]);
    const amax = d3.max(inView.length ? inView : rows, (r) => Math.abs(r.amount)) || 100;
    const y = d3.scaleSymlog().constant(10).domain([Math.min(0, d3.min(rows, (r) => r.amount)), amax * 1.1]).range([H1[1], H1[0]]);
    const yr = d3.scaleLinear().domain([0, 1]).range([H2[1], H2[0]]);
    const clip = `clip-${Math.random().toString(36).slice(2)}`;
    svg.append("clipPath").attr("id", clip).append("rect").attr("x", m.l).attr("y", 0).attr("width", w - m.l - m.r).attr("height", H2[1] + 2);

    // grid + axes
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues([0, 10, 100, 1000].filter((v) => v <= amax * 1.1)).tickSize(-(w - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues([0, 10, 100, 1000].filter((v) => v <= amax * 1.1)).tickFormat((v) => `$${v}`).tickSize(0)).select(".domain").remove();
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H1[1] + 2})`).call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(w / 110))).tickSize(3));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(yr).ticks(2).tickSize(0).tickFormat((v) => v.toFixed(1))).select(".domain").remove();
    svg.append("text").attr("x", 4).attr("y", (H2[0] + H2[1]) / 2).text("RISK");
    svg.append("text").attr("x", 4).attr("y", H1[0] + 10).text("AMOUNT");

    const g = svg.append("g").attr("clip-path", `url(#${clip})`);
    // threshold band in risk panel
    g.append("rect").attr("x", m.l).attr("width", w - m.l - m.r).attr("y", yr(1)).attr("height", yr(thr) - yr(1)).attr("fill", cssVar("--bad")).attr("opacity", 0.07);
    g.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", yr(thr)).attr("y2", yr(thr)).attr("stroke", cssVar("--bad")).attr("stroke-dasharray", "4 3");
    g.append("path").datum(rows).attr("fill", "none").attr("stroke", cssVar("--phos")).attr("stroke-width", 1.5)
      .attr("d", d3.line().curve(d3.curveStepAfter).x((r) => x(new Date(r.ts * 1000))).y((r) => yr(r.score)));
    g.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", H2[1]).attr("y2", H2[1]).attr("stroke", cssVar("--line-2"));

    // transactions
    const legit = rows.filter((r) => !r.is_fraud), fraud = rows.filter((r) => r.is_fraud);
    const flagged = rows.filter((r) => r.score >= thr);
    g.selectAll(".lg").data(legit).join("circle").attr("cx", (r) => x(new Date(r.ts * 1000))).attr("cy", (r) => y(r.amount)).attr("r", 2.6)
      .attr("fill", cssVar("--dim")).attr("opacity", 0.75);
    g.selectAll(".fl").data(flagged).join("circle").attr("cx", (r) => x(new Date(r.ts * 1000))).attr("cy", (r) => y(r.amount)).attr("r", 7)
      .attr("fill", "none").attr("stroke", cssVar("--phos")).attr("stroke-width", 1.5);
    g.selectAll(".fr").data(fraud).join("path").attr("d", d3.symbol(d3.symbolTriangle, 70))
      .attr("transform", (r) => `translate(${x(new Date(r.ts * 1000))},${y(r.amount)})`)
      .attr("fill", cssVar("--bad")).attr("stroke", cssVar("--bg-2")).attr("stroke-width", 1.5);
    if (focus) {
      g.append("circle").attr("cx", x(new Date(focus.ts * 1000))).attr("cy", y(focus.amount)).attr("r", 11).attr("fill", "none")
        .attr("stroke", cssVar("--text")).attr("stroke-width", 1);
      g.append("line").attr("x1", x(new Date(focus.ts * 1000))).attr("x2", x(new Date(focus.ts * 1000))).attr("y1", H1[0]).attr("y2", H2[1])
        .attr("stroke", cssVar("--text")).attr("opacity", 0.25);
    }
    // hit targets (bigger than the marks), nearest point by voronoi
    const pts = inView.map((r) => [x(new Date(r.ts * 1000)), y(r.amount), r]);
    if (pts.length) {
      const del = d3.Delaunay.from(pts, (p) => p[0], (p) => p[1]);
      svg.append("rect").attr("x", m.l).attr("y", H1[0]).attr("width", w - m.l - m.r).attr("height", H1[1] - H1[0]).attr("fill", "transparent").style("cursor", "crosshair")
        .on("mousemove", (ev) => {
          const [mx, my] = d3.pointer(ev);
          const p = pts[del.find(mx, my)];
          if (!p || Math.hypot(p[0] - mx, p[1] - my) > 24) { tip.hide(); return; }
          const r = p[2];
          tip.show(ev, `<b>${tsFull(r.ts)}</b><br>${fmtMoney(r.amount)} · ${upper(place(r))}<br>${mccName(r.mcc)} · ${upper(r.channel)}<br>score <b>${r.score.toFixed(3)}</b> ${r.score >= thr ? "▲ flagged" : "· passed"}<br>${r.is_fraud ? '<span style="color:var(--bad)">▲ FRAUD</span>' : '<span style="color:var(--ok)">✓ legit</span>'}`);
        })
        .on("mouseleave", () => tip.hide())
        .on("click", (ev) => {
          const [mx, my] = d3.pointer(ev);
          const p = pts[del.find(mx, my)];
          if (p && Math.hypot(p[0] - mx, p[1] - my) <= 24) onPick(p[2]);
        });
    }

    // context strip + brush
    const yc = d3.scaleLinear().domain([0, 1]).range([H3[1], H3[0]]);
    svg.append("rect").attr("x", m.l).attr("y", H3[0]).attr("width", w - m.l - m.r).attr("height", H3[1] - H3[0]).attr("fill", cssVar("--bg"));
    svg.selectAll(".cx").data(rows).join("rect").attr("x", (r) => xc(new Date(r.ts * 1000))).attr("width", 1)
      .attr("y", (r) => (r.is_fraud ? H3[0] : yc(0.5))).attr("height", (r) => (r.is_fraud ? H3[1] - H3[0] : (H3[1] - H3[0]) / 2))
      .attr("fill", (r) => (r.is_fraud ? cssVar("--bad") : cssVar("--line-2")));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H3[1]})`).call(d3.axisBottom(xc).ticks(6).tickSize(2)).select(".domain").remove();
    const brush = d3.brushX().extent([[m.l, H3[0]], [w - m.r, H3[1]]]).on("end", (ev) => {
      if (!ev.sourceEvent) return;
      if (!ev.selection) { setDomain(null); return; }
      const [a, b] = ev.selection.map((v) => xc.invert(v).getTime() / 1000);
      if (b - a > 3600) setDomain([a, b]);
    });
    const bg = svg.append("g").call(brush);
    bg.call(brush.move, x.domain().map((d) => xc(d)));
    bg.selectAll(".selection").attr("fill", cssVar("--phos")).attr("fill-opacity", 0.12).attr("stroke", cssVar("--phos"));
  }, [tl, domain, focus]);
  return html`<div class="chart-box" style="height:420px" ref=${ref}></div>`;
}
