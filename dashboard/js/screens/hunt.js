import { html, useState, useEffect, useApp, api, Pane, ScoreBar, fmtMoney, fmtPct, tsFull, ago, place, upper, mccName,
  shortModel, modelColor } from "../lib.js";

const ROUNDS = 10;

export function Hunt() {
  const { split, model, setStatus, keys } = useApp();
  const [seed0] = useState(() => Math.floor(Math.random() * 1e6));
  const [game, setGame] = useState(0);
  const [i, setI] = useState(0);
  const [round, setRound] = useState(null);
  const [guess, setGuess] = useState(null);
  const [rev, setRev] = useState(null);
  const [log, setLog] = useState([]);
  const over = log.length >= ROUNDS && rev;

  useEffect(() => {
    setRound(null); setGuess(null); setRev(null);
    api("/api/game/round", { seed: seed0 + game * 100 + i, model }, split).then(setRound);
  }, [i, game]);

  const choose = async (g) => {
    if (!round || guess != null) return;
    setGuess(g);
    const r = await api(`/api/game/reveal/${round.round}`, {}, split);
    setRev(r);
    const ms = r.scores.find((x) => x.model === model);
    setLog((l) => [...l, { you: (g === 1) === !!r.is_fraud, model: ms.flagged === !!r.is_fraud, fraud: r.is_fraud }]);
  };
  const next = () => {
    if (!rev) return;
    if (log.length >= ROUNDS) { setLog([]); setGame((x) => x + 1); setI(0); return; }
    setI((x) => x + 1);
  };

  useEffect(() => {
    keys.current = (e) => {
      const k = e.key.toLowerCase();
      if (k === "f") { choose(1); return true; }
      if (k === "l") { choose(0); return true; }
      if (k === "n" || k === "enter") { next(); return true; }
      return false;
    };
    return () => { keys.current = null; };
  });

  const you = log.filter((x) => x.you).length, mdl = log.filter((x) => x.model).length;
  useEffect(() => { setStatus(`ROUND ${Math.min(log.length + (rev ? 0 : 1), ROUNDS)}/${ROUNDS} · YOU ${you} · ${shortModel(model)} ${mdl}`); }, [log, rev, model]);

  const t = round?.transaction;
  return html`<div class="grid g-hunt">
    <div class="stack">
      <${Pane} title=${`CASE ${t ? t.txn_id : "…"} · CARD u${t?.user ?? "…"}·c${t?.card ?? "…"}`} right=${`round ${Math.min(log.length + (rev ? 0 : 1), ROUNDS)} of ${ROUNDS}`}>
        ${!t ? html`<div class="muted blink">pulling a transaction…</div>` : html`
        <div class="record">
          ${rev ? html`<div class=${`stamp ${rev.is_fraud ? "bad" : "ok"}`}>${rev.is_fraud ? "FRAUD" : "LEGIT"}</div>` : null}
          <div class="muted" style="font-size:11px;letter-spacing:.12em">${tsFull(t.ts)} · ${upper(t.channel)} TRANSACTION</div>
          <div class="amt">${t.amount < 0 ? "−" : ""}${fmtMoney(t.amount)}</div>
          <div class="where">${upper(place(t))}</div>
          <div class="meta">
            <span>MCC ${t.mcc} · ${mccName(t.mcc)}</span><span>merchant …${t.merchant}</span>
            ${t.errors ? html`<span class="bad">ERROR: ${upper(t.errors)}</span>` : html`<span class="muted">no errors</span>`}
          </div>
          <div class="choice">
            <button class="btn bad huge" disabled=${guess != null} onClick=${() => choose(1)}><kbd>F</kbd>FRAUD</button>
            <button class="btn ok huge" disabled=${guess != null} onClick=${() => choose(0)}><kbd>L</kbd>LEGIT</button>
            ${rev ? html`<button class="btn pri huge" onClick=${next}><kbd>N</kbd>${over ? "PLAY AGAIN" : "NEXT CASE"}</button>` : null}
          </div>
        </div>`}
      </${Pane}>

      <${Pane} cls="flush" title="CARD HISTORY · PREVIOUS 6" right="what the card did before this purchase">
        <table class="tbl">
          <thead><tr><th>BEFORE</th><th class="num">AMOUNT</th><th>LOCATION</th><th>CATEGORY</th><th>CHAN</th></tr></thead>
          <tbody>${(round?.history || []).map((h) => html`<tr>
            <td class="muted">${ago(t.ts - h.ts)} earlier</td><td class="num">${fmtMoney(h.amount)}</td>
            <td>${upper(place(h)).slice(0, 26)}</td><td class="muted">${mccName(h.mcc)}</td><td class="muted">${upper(h.channel).slice(0, 6)}</td></tr>`)}
            ${round && !round.history.length ? html`<tr><td colspan="5" class="muted">first transaction we've seen on this card</td></tr>` : null}</tbody>
        </table>
      </${Pane}>
    </div>

    <div class="stack">
      <${Pane} title="SCOREBOARD" right="this deck is ~45% fraud; real traffic is 0.14%">
        <div class="score-big">
          <div><div class="who">YOU</div><div class="pts phos">${you}<span class="muted">/${log.length}</span></div>
            <div class="pips">${log.map((x) => html`<span class=${`pip ${x.you ? "w" : "l"}`}></span>`)}</div></div>
          <div><div class="who">${shortModel(model)}</div><div class="pts" style=${`color:${modelColor(model)}`}>${mdl}<span class="muted">/${log.length}</span></div>
            <div class="pips">${log.map((x) => html`<span class=${`pip ${x.model ? "w" : "l"}`}></span>`)}</div></div>
        </div>
        ${over ? html`<p style="margin:14px 0 0" class=${you > mdl ? "ok" : you === mdl ? "phos" : "bad"}>
          ${you > mdl ? "▲ You beat the model. Fraud teams are hiring." : you === mdl ? "■ A draw. Not bad against 24M transactions of training." : "▼ The model wins this time. Press N for a rematch."}</p>` : null}
      </${Pane}>

      <${Pane} title="MODEL VERDICTS" right=${rev ? "the tick is each model's threshold" : "hidden until you decide"}>
        ${!rev ? html`<div class="muted">Commit to FRAUD or LEGIT to see what the four models said.</div>` : html`
          <table class="tbl"><tbody>${rev.scores.map((s) => html`<tr>
            <td><span style=${`color:${modelColor(s.model)}`}>■</span> ${shortModel(s.model)}</td>
            <td><${ScoreBar} score=${s.score} thr=${s.threshold} width=${16} /></td>
            <td class="num">${s.score.toFixed(3)}</td>
            <td>${s.flagged ? html`<span class="v bad">▲ FLAG</span>` : html`<span class="v ok">✓ PASS</span>`}</td>
            <td>${s.flagged === !!rev.is_fraud ? html`<span class="ok">right</span>` : html`<span class="bad">wrong</span>`}</td></tr>`)}</tbody></table>`}
      </${Pane}>

      ${rev?.reasons ? html`<${Pane} title="WHY · TOP SHAP REASONS" right=${shortModel(rev.reasons.model)}>
        ${(() => {
          const mx = Math.max(...rev.reasons.items.map((r) => Math.abs(r.shap)));
          return rev.reasons.items.map((r) => html`<div class="reason">
            <span>${r.feature} = <b>${r.value ?? "missing"}</b></span>
            <span style="display:flex;align-items:center;gap:6px"><span class="rb" style=${`width:${(Math.abs(r.shap) / mx) * 80}px;background:${r.shap > 0 ? "var(--bad)" : "var(--ok)"}`}></span>
            <span class=${r.shap > 0 ? "bad" : "ok"}>${r.shap > 0 ? "▲" : "▼"}${Math.abs(r.shap).toFixed(2)}</span></span></div>`);
        })()}
        <div class="muted" style="font-size:11px;margin-top:6px">▲ pushes toward fraud · ▼ toward legit (log-odds)</div>
      </${Pane}>` : null}
    </div>
  </div>`;
}
