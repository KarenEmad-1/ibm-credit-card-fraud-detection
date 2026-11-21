import { html, useState, useEffect, useRef, useApp, Pane, ScoreBar, Verdict, Tween, useD3, d3, tip,
  fmtInt, fmtPct, fmtMoney, fmtK, tsFull, tsTime, place, upper, cssVar, Key } from "../lib.js";

const SPEEDS = [{ v: 30, l: "30m/s" }, { v: 120, l: "2h/s" }, { v: 360, l: "6h/s" }, { v: 1440, l: "1d/s" }, { v: 4320, l: "3d/s" }];
const TAPE_MAX = 34;
const DELAYS = [0, 3, 14, 45];

export function Tape() {
  const { meta, model, modelMeta, split, setStatus, keys } = useApp();
  const [playing, setPlaying] = useState(true);
  const [speedI, setSpeedI] = useState(2);
  const [delay, setDelay] = useState(3);
  const [s, setS] = useState(() => fresh(meta));
  const es = useRef(null);
  const simRef = useRef(s.sim);

  // reset first, so the stream below starts from the new model's first day
  useEffect(() => { const f = fresh(meta); simRef.current = f.sim; setS(f); }, [model]);

  // ---------------------------------------------------------------- stream
  useEffect(() => {
    if (!playing) return;
    const q = new URLSearchParams({ model, speed: SPEEDS[speedI].v, split, start: new Date(simRef.current * 1000).toISOString().slice(0, 19) });
    const src = new EventSource(`/api/replay?${q}`);
    es.current = src;
    src.addEventListener("tick", (ev) => {
      const d = JSON.parse(ev.data);
      simRef.current = d.sim_time;
      setS((p) => {
        const hourKey = Math.floor(d.sim_time / 3600);
        const hours = p.hours.length && p.hours[p.hours.length - 1].h === hourKey
          ? [...p.hours.slice(0, -1), { h: hourKey, alerts: p.hours[p.hours.length - 1].alerts + d.alerts, txns: p.hours[p.hours.length - 1].txns + d.transactions }]
          : [...p.hours, { h: hourKey, alerts: d.alerts, txns: d.transactions }].slice(-96);
        const fresh = d.tape_rows.map((r) => ({ ...r, isNew: true })).sort((a, b) => b.ts - a.ts);
        return {
          ...p, sim: d.sim_time, txns: p.txns + d.transactions, volume: p.volume + d.amount, nAlerts: p.nAlerts + d.alerts,
          tape: [...fresh, ...p.tape.map((r) => ({ ...r, isNew: false }))].slice(0, TAPE_MAX),
          alerts: [...d.alert_rows.slice().reverse(), ...p.alerts], hours,
        };
      });
    });
    src.addEventListener("day", (ev) => { const d = JSON.parse(ev.data); setS((p) => ({ ...p, days: [...p.days, d] })); });
    src.addEventListener("end", () => setPlaying(false));
    src.onerror = () => { src.close(); };
    return () => src.close();
  }, [playing, speedI, model, split]);

  // ---------------------------------------------------------------- keys + status
  useEffect(() => {
    keys.current = (e) => {
      if (e.key === " ") { setPlaying((x) => !x); return true; }
      if (e.key === "+" || e.key === "=") { setSpeedI((i) => Math.min(SPEEDS.length - 1, i + 1)); return true; }
      if (e.key === "-" || e.key === "_") { setSpeedI((i) => Math.max(0, i - 1)); return true; }
      return false;
    };
    return () => { keys.current = null; };
  }, []);
  useEffect(() => {
    setStatus(html`<span class=${playing ? "phos" : "muted"}>${playing ? "● LIVE" : "❚❚ PAUSED"}</span>  SIM ${tsFull(s.sim)}  ×${SPEEDS[speedI].l}`);
  }, [s.sim, playing, speedI]);

  // ---------------------------------------------------------------- derived (labels arrive after `delay` days)
  const labelled = (ts) => s.sim >= ts + delay * 86400;
  const lab = s.alerts.filter((a) => labelled(a.ts));
  const tp = lab.filter((a) => a.is_fraud);
  const stopped = tp.reduce((t, a) => t + Math.abs(a.amount), 0);
  const doneDays = s.days.filter((d) => s.sim >= Date.parse(`${d.date}T00:00:00Z`) / 1000 + 86400 + delay * 86400);
  const dFr = doneDays.reduce((t, d) => t + d.frauds, 0), dTp = doneDays.reduce((t, d) => t + d.true_alerts, 0);
  const fraudAmt = doneDays.reduce((t, d) => t + d.fraud_amount, 0);
  const lastPsi = s.days.length ? s.days[s.days.length - 1].psi : {};

  return html`<div class="grid g-tape">
    <div class="stack">
      <div class="kpis">
        ${kpi("TXN SCORED", html`<${Tween} value=${s.txns} format=${fmtInt} ms=${400} />`, fmtK(s.volume) + " volume")}
        ${kpi("ALERTS", html`<${Tween} value=${s.nAlerts} format=${fmtInt} ms=${400} />`, `${fmtPct(s.txns ? s.nAlerts / s.txns : 0, 3)} of traffic`, "bad")}
        ${kpi("PRECISION", lab.length ? fmtPct(tp.length / lab.length) : "—", `${tp.length}/${lab.length} labelled alerts`)}
        ${kpi("RECALL", dFr ? fmtPct(dTp / dFr) : "—", `${fmtInt(dFr - dTp)} frauds slipped through`)}
        ${kpi("$ STOPPED", fmtK(stopped), dFr ? `of ${fmtK(fraudAmt)} fraud` : "awaiting labels", "ok")}
      </div>
      <${Pane} cls="flush" title="TRANSACTION TAPE"
        right=${html`<span class="ctl">
          <${Key} k="SPACE" onClick=${() => setPlaying((x) => !x)}>${playing ? "PAUSE" : "PLAY"}</${Key}>
          <${Key} k="−" onClick=${() => setSpeedI((i) => Math.max(0, i - 1))}></${Key}>
          <span class="phos">${SPEEDS[speedI].l}</span>
          <${Key} k="+" onClick=${() => setSpeedI((i) => Math.min(SPEEDS.length - 1, i + 1))}></${Key}>
        </span>`}>
        <div class="scroll" style="max-height: 560px">
          <table class="tbl">
            <thead><tr><th>TIME</th><th>CARD</th><th class="num">AMOUNT</th><th>LOCATION</th><th>CHAN</th><th>RISK</th><th class="num">SCORE</th><th></th></tr></thead>
            <tbody>${s.tape.length ? s.tape.map((r) => {
              const hot = r.score >= modelMeta.threshold;
              return html`<tr key=${r.txn_id} class=${`${hot ? "alert" : ""} ${hot && r.isNew ? "flash" : ""}`}>
                <td class="muted">${tsTime(r.ts)}</td><td>u${r.user}·c${r.card}</td>
                <td class="num">${fmtMoney(r.amount)}</td><td>${upper(place(r)).slice(0, 26)}</td>
                <td class="muted">${upper(r.channel).slice(0, 6)}</td>
                <td><${ScoreBar} score=${r.score} thr=${modelMeta.threshold} /></td>
                <td class="num ${hot ? "bad" : "muted"}">${r.score.toFixed(3)}</td>
                <td>${hot ? html`<span class="v bad">▲ ALERT</span>` : html`<span class="muted">· pass</span>`}</td></tr>`;
            }) : html`<tr><td colspan="8" class="muted blink">waiting for transactions…</td></tr>`}</tbody>
          </table>
        </div>
      </${Pane}>
    </div>

    <div class="stack">
      <${Pane} title="ALERTS / HOUR" right="last 96 sim-hours">
        <${Spark} hours=${s.hours} />
      </${Pane}>
      <${Pane} cls="flush" title="ALERT QUEUE" right=${html`<span class="ctl">LABEL DELAY
        <select class="inp" value=${delay} onChange=${(e) => setDelay(+e.target.value)}>
          ${DELAYS.map((d) => html`<option value=${d}>${d}d</option>`)}</select></span>`}>
        <div class="scroll" style="max-height: 300px">
          <table class="tbl">
            <thead><tr><th>RAISED</th><th>CARD</th><th class="num">AMOUNT</th><th>WHERE</th><th>STATUS</th></tr></thead>
            <tbody>${s.alerts.slice(0, 40).map((a) => html`<tr key=${a.txn_id}>
              <td class="muted">${tsFull(a.ts).slice(5)}</td><td>u${a.user}·c${a.card}</td>
              <td class="num">${fmtMoney(a.amount)}</td><td>${upper(place(a)).slice(0, 18)}</td>
              <td><${Verdict} fraud=${a.is_fraud} pending=${!labelled(a.ts)} /></td></tr>`)}
              ${!s.alerts.length ? html`<tr><td colspan="5" class="muted">no alerts yet</td></tr>` : null}</tbody>
          </table>
        </div>
      </${Pane}>
      <${Pane} title="FEATURE DRIFT · PSI" right=${`7-day window vs ${meta.drift.reference}`}>
        ${meta.drift.features.map((f) => {
          const v = lastPsi[f];
          const w = v == null ? 0 : Math.min(1, v / 0.3);
          return html`<div class="meter" key=${f}><span class="muted">${f}</span>
            <span class="track"><span class=${`fill ${v >= 0.25 ? "bad" : v >= 0.1 ? "warn" : ""}`} style=${`width:${w * 100}%`}></span>
              <span class="mark" style="left:33.3%"></span><span class="mark" style="left:83.3%"></span></span>
            <span class="right ${v >= 0.25 ? "bad" : ""}">${v == null ? "—" : v.toFixed(3)}</span></div>`;
        })}
        <div class="muted" style="font-size:11px;margin-top:6px">marks at 0.10 (watch) and 0.25 (investigate)</div>
      </${Pane}>
    </div>
  </div>`;
}

function fresh(meta) {
  return { sim: Date.parse(`${meta.year}-01-01T00:00:00Z`) / 1000, txns: 0, volume: 0, nAlerts: 0, tape: [], alerts: [], hours: [], days: [] };
}

const kpi = (l, n, sub, cls = "") => html`<div class="kpi"><div class="l">${l}</div><div class=${`n ${cls}`}>${n}</div><div class="s">${sub}</div></div>`;

function Spark({ hours }) {
  const ref = useD3((el, w, h) => {
    const svg = d3.select(el).append("svg").attr("class", "chart").attr("width", w).attr("height", h);
    if (!hours.length) { svg.append("text").attr("x", 8).attr("y", h / 2).text("—"); return; }
    const m = { t: 6, r: 6, b: 16, l: 26 };
    const x = d3.scaleLinear().domain([hours[0].h, Math.max(hours[0].h + 95, hours[hours.length - 1].h)]).range([m.l, w - m.r]);
    const y = d3.scaleLinear().domain([0, Math.max(3, d3.max(hours, (d) => d.alerts))]).nice().range([h - m.b, m.t]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickSize(-(w - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(3).tickSize(0)).select(".domain").remove();
    const bw = Math.max(1, (w - m.l - m.r) / 96 - 1);
    svg.selectAll("rect").data(hours).join("rect")
      .attr("x", (d) => x(d.h)).attr("width", bw).attr("y", (d) => y(d.alerts)).attr("height", (d) => y(0) - y(d.alerts))
      .attr("fill", cssVar("--bad")).attr("rx", 1)
      .on("mousemove", (ev, d) => tip.show(ev, `<b>${new Date(d.h * 3600000).toISOString().slice(0, 13).replace("T", " ")}h</b><br>${d.alerts} alerts · ${fmtInt(d.txns)} txns`))
      .on("mouseleave", () => tip.hide());
    svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", cssVar("--line-2"));
  }, [hours]);
  return html`<div class="chart-box short" ref=${ref}></div>`;
}
