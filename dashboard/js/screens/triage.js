import { html, useState, useEffect, useApp, api, Pane, ScoreBar, Verdict, fmtMoney, fmtInt, tsTime, place, upper, mccName,
  shortModel } from "../lib.js";

const K = 24;

export function Triage() {
  const { split, model, modelMeta, setStatus, keys } = useApp();
  const [days, setDays] = useState([]);
  const [date, setDate] = useState(null);
  const [budget, setBudget] = useState(6);
  const [q, setQ] = useState(null);
  const [picks, setPicks] = useState(new Set());
  const [rev, setRev] = useState(null);

  useEffect(() => { api("/api/triage/days", { n: 14, seed: 7 }, split).then((r) => { setDays(r.days); setDate(r.days[0]); }); }, []);
  useEffect(() => {
    if (!date) return;
    setQ(null); setRev(null); setPicks(new Set());
    api("/api/triage", { date, model, k: K }, split).then(setQ);
  }, [date, model]);

  const toggle = (id) => {
    if (rev) return;
    setPicks((p) => { const n = new Set(p); n.has(id) ? n.delete(id) : n.size < budget && n.add(id); return n; });
  };
  const submit = async () => {
    if (!q || rev || !picks.size) return;
    setRev(await api("/api/triage/reveal", { date, model, k: K }, split));
  };
  const nextDay = () => { const i = days.indexOf(date); setDate(days[(i + 1) % days.length]); };

  useEffect(() => {
    keys.current = (e) => {
      if (e.key === "Enter") { submit(); return true; }
      if (e.key.toLowerCase() === "r") { nextDay(); return true; }
      return false;
    };
    return () => { keys.current = null; };
  });
  useEffect(() => { setStatus(`SHIFT ${date || "…"} · ${picks.size}/${budget} INVESTIGATIONS USED`); }, [date, picks, budget]);

  const strategies = rev && q ? rank(q.queue, rev.labels, picks, budget, model) : null;
  const best = strategies ? Math.max(...strategies.map((s) => s.amount), 1) : 1;

  return html`<div class="grid g-2">
    <${Pane} cls="flush" title=${`SHIFT QUEUE · ${date || "…"}`} right=${q ? `top ${K} of ${fmtInt(q.day_transactions)} txns by ${shortModel(model)} score` : ""}>
      <div class="ctl" style="padding:8px 10px;border-bottom:1px solid var(--line)">
        SHIFT <select class="inp" value=${date} onChange=${(e) => setDate(e.target.value)}>${days.map((d) => html`<option value=${d}>${d}</option>`)}</select>
        BUDGET <input class="inp" type="range" min="3" max="12" value=${budget} disabled=${!!rev}
          onInput=${(e) => { setBudget(+e.target.value); setPicks(new Set()); }} />
        <b>${budget}</b> investigations
        <span class="grow"></span>
        <button class="btn pri" disabled=${!picks.size || !!rev} onClick=${submit}><kbd>⏎</kbd>CLOSE SHIFT</button>
        <button class="btn" onClick=${nextDay}><kbd>R</kbd>NEXT SHIFT</button>
      </div>
      <table class="tbl click">
        <thead><tr><th></th><th>TIME</th><th class="num">AMOUNT</th><th>LOCATION</th><th>CATEGORY</th><th>RISK</th><th class="num">SCORE</th>${rev ? html`<th>TRUTH</th>` : null}</tr></thead>
        <tbody>${(q?.queue || []).map((r) => {
          const on = picks.has(r.txn_id);
          return html`<tr class=${on ? "picked" : ""} onClick=${() => toggle(r.txn_id)}>
            <td class="pick">${on ? "[■]" : "[ ]"}</td><td class="muted">${tsTime(r.ts)}</td>
            <td class="num">${fmtMoney(r.amount)}</td><td>${upper(place(r)).slice(0, 22)}</td><td class="muted">${mccName(r.mcc)}</td>
            <td><${ScoreBar} score=${r.score} thr=${modelMeta.threshold} width=${10} /></td><td class="num">${r.score.toFixed(3)}</td>
            ${rev ? html`<td><${Verdict} fraud=${rev.labels[r.txn_id]} /></td>` : null}</tr>`;
        })}
        ${!q ? html`<tr><td colspan="8" class="muted blink">building queue…</td></tr>` : null}</tbody>
      </table>
    </${Pane}>

    <div class="stack">
      <${Pane} title="YOUR SHIFT" right="click rows to investigate them">
        <div class="prose">
          <p>You have <b>${budget}</b> investigations for this shift. The queue holds the day's ${K} riskiest transactions by model score, and most of them are probably fine.</p>
          <p>The ranking question: work the <b>highest score</b>, or the <b>most money at risk</b> (score × amount)? Investigating a fraud recovers its amount.</p>
        </div>
      </${Pane}>
      <${Pane} title="SHIFT REPORT" right=${rev ? `${rev.day_frauds} frauds that day · ${fmtMoney(rev.day_fraud_amount)}` : "close the shift to score it"}>
        ${!strategies ? html`<div class="muted">Pick up to ${budget} transactions, then press ⏎ CLOSE SHIFT.</div>` : html`
          ${strategies.map((s) => html`<div class=${`strat ${s.you ? "you" : ""}`}>
            <span class=${s.you ? "phos" : ""}>${s.name}</span>
            <span><span class="bar" style=${`display:block;width:${(s.amount / best) * 100}%;min-width:2px;${s.you ? "" : s.hind ? "background:var(--ok)" : ""}`}></span></span>
            <span class="right">${fmtMoney(s.amount, 0)} · ${s.n}</span></div>`)}
          <p class="prose" style="margin-top:10px">${verdict(strategies)}</p>
          <p class="muted" style="font-size:11px">Amount = fraud dollars recovered · count = frauds caught. Hindsight knows the labels, so it's the ceiling.</p>`}
      </${Pane}>
    </div>
  </div>`;
}

function rank(queue, labels, picks, budget, model) {
  const take = (arr) => arr.slice(0, budget);
  const score = (rows, name, extra = {}) => {
    const fr = rows.filter((r) => labels[r.txn_id]);
    return { name, amount: fr.reduce((t, r) => t + Math.abs(r.amount), 0), n: fr.length, ...extra };
  };
  return [
    score(queue.filter((r) => picks.has(r.txn_id)), "YOU", { you: true }),
    score(take([...queue].sort((a, b) => b.score - a.score)), `${shortModel(model)} RANK`),
    score(take([...queue].sort((a, b) => b.score * Math.abs(b.amount) - a.score * Math.abs(a.amount))), "SCORE × AMOUNT"),
    score(take([...queue].sort((a, b) => Math.abs(b.amount) - Math.abs(a.amount))), "BIGGEST $ FIRST"),
    score(take(queue.filter((r) => labels[r.txn_id]).sort((a, b) => Math.abs(b.amount) - Math.abs(a.amount))), "HINDSIGHT", { hind: true }),
  ];
}

function verdict(s) {
  const [you, mdl, ev] = s;
  if (you.amount > Math.max(mdl.amount, ev.amount)) return "▲ You beat every ranking rule. Good instincts.";
  if (ev.amount > mdl.amount) return "Ranking by score × amount recovers more than the raw score. A good model is necessary but not sufficient: the ranking rule matters too.";
  if (you.amount === mdl.amount) return "■ You matched the model's ranking exactly.";
  return "▼ The model's ranking recovered more. Try another shift.";
}
