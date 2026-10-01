/* Copyright SZL Holdings LLC. SPDX-License-Identifier: Apache-2.0 */
'use strict';
const byId = id => document.getElementById(id);
const familyNotes = {
  clean: 'Both channels follow their ordinary low-noise behavior.',
  shared_bias: 'Cheap observations share one hidden bias. Repeating a source can repeat a mistake.',
  independent_outliers: 'The cheap channel contains more independent observation errors.',
  reference_outliers: 'The reference channel is deliberately noisier. A separate source can still be wrong.',
  bias_drift: 'Cheap-source bias changes during the episode, challenging the fixed-bias assumptions.',
  compositional: 'Two groups change independently, challenging the structured world assumptions.'
};
const policyNotes = {
  learned: 'An original trained neural selector chooses observations. The belief update uses the supplied Bayesian model.',
  source_aware: 'A programmed exact one-step Bayesian acquisition rule models the shared source bias.',
  independent_bias: 'An ablation treats cheap-source errors as independent across reports.',
  no_confirmation: 'Retain the four inherited reports without buying another observation.',
  cheap_only: 'Buy only from the cheaper channel.',
  reference_only: 'Buy only from the separate reference channel.',
  random_mixed: 'Choose affordable observations from either channel at random.'
};
const human = value => String(value).replaceAll('_', ' ');
const pct = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '—';
const number = value => Number.isFinite(value) ? value.toFixed(4) : '—';
let current = null;
let currentRaw = null;
let running = false;
let truth = false;
let ready = false;
let retryTimer = null;
const ownTrials = [];
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function api(path, options) {
  const response = await fetch(path, options);
  const raw = await response.text();
  let data;
  try { data = JSON.parse(raw); } catch (_) { throw new Error('The workbench returned no readable record.'); }
  if (!response.ok) {
    const failureReceipt = response.status === 503 && data.schema === 'szl.confirmation.public-exploratory-receipt/v1'
      && data.status === 'FAILED' && /^[0-9a-f]{32}$/.test(data.id) && /^[0-9a-f]{64}$/.test(data.receipt_sha256)
      && data.request && typeof data.request.family === 'string' && typeof data.request.policy === 'string';
    if (failureReceipt) return {data, raw};
    const error = new Error(data.error || `Trial refused (${response.status}).`);
    error.retry = response.status === 429 ? Math.max(1, Math.min(60, Number(response.headers.get('retry-after')) || 5)) : 0;
    throw error;
  }
  return {data, raw};
}
function descriptions() {
  byId('family-description').textContent = familyNotes[byId('family').value];
  byId('policy-description').textContent = policyNotes[byId('policy').value];
  byId('model-seed').disabled = byId('policy').value !== 'learned';
}
async function refresh() {
  try {
    const {data} = await api('/api/status');
    ready = data.ready === true && data.models_loaded?.length === 3 && data.startup_execution_probes?.length === 3;
    byId('connection').textContent = ready ? 'Ready for a fresh trial' : 'Workbench unavailable';
    byId('connection-dot').className = ready ? 'dot ready' : 'dot';
    byId('sources').textContent = ready ? 'Ready' : data.state === 'STARTING' ? 'Starting' : 'Unavailable';
    byId('checkpoints').textContent = String(data.checkpoints_verified || 0);
    byId('loaded').textContent = `${data.startup_execution_probes?.length || 0} / 3`;
    byId('status-message').textContent = ready
      ? data.busy ? 'Another trial is running. Check availability shortly to start yours.' : 'The frozen release is verified. All three trained checkpoints executed a startup check. Your trial runs new inference.'
      : data.error || 'The free CPU workbench is starting. Retry shortly; no results are fabricated while it is unavailable.';
    byId('run-trial').disabled = running || !ready || data.busy || retryTimer !== null;
    byId('retention-message').textContent = 'Server receipts: ephemeral memory, at most 24 hours and 128 records, lost on restart. This page keeps only the receipt IDs from your current visit.';
  } catch (_) {
    ready = false;
    byId('connection').textContent = 'Workbench unavailable';
    byId('connection-dot').className = 'dot';
    byId('status-message').textContent = 'The on-demand workbench may be waking or unavailable. Reload to check again.';
    byId('run-trial').disabled = true;
  }
}
function sessionList() {
  const target = byId('session-trials'); target.replaceChildren();
  for (const receipt of ownTrials) {
    const button = element('button', `${human(receipt.request.family)} · ${human(receipt.request.policy)} · ${receipt.id.slice(0, 8)}`, 'recent-item');
    button.addEventListener('click', async () => {
      try { const result = await api(`/api/trials/${receipt.id}`); render(result.data, result.raw); }
      catch (error) { byId('trial-message').textContent = error.message; }
    });
    target.append(button);
  }
}
function render(receipt, raw) {
  current = receipt; currentRaw = raw; truth = false;
  byId('export').disabled = false;
  byId('trial-heading').textContent = `${human(receipt.request.family)} / ${human(receipt.request.policy)}`;
  byId('receipt-json').textContent = JSON.stringify(receipt, null, 2);
  byId('receipt-details').hidden = false;
  const result = receipt.result;
  for (const id of ['score-cards', 'prediction-section', 'history-section']) byId(id).hidden = !result;
  if (!result) { byId('trial-message').textContent = 'This trial failed. Its unsigned failure record is available to download.'; return; }
  byId('trial-message').textContent = `Fresh exploratory inference completed ${new Date(receipt.completed_at).toLocaleString()} · world ${receipt.request.seed} / episode ${receipt.request.index}.`;
  const cards = byId('score-cards'); cards.replaceChildren();
  for (const [title, value] of [['Weighted accuracy', pct(result.accuracy)], ['Accuracy − cost', number(result.net_utility)], ['Confident error', pct(result.confident_wrong)], ['Paid observation cost', number(result.paid_cost)]]) {
    const card = element('div', undefined, 'score'); card.append(element('span', title), element('strong', value)); cards.append(card);
  }
  byId('step').max = String(result.history.length); byId('step').value = String(result.history.length);
  replay(); predictions();
}
function replay() {
  if (!current?.result) return;
  const history = current.result.history;
  const count = Math.max(0, Math.min(history.length, Number(byId('step').value)));
  byId('step-output').textContent = `${count} / ${history.length}`;
  byId('previous-step').disabled = count === 0;
  byId('next-step').disabled = count === history.length;
  const observation = history[count - 1];
  byId('current-observation').textContent = observation
    ? `${human(observation.source)} · group ${observation.group} · observed label ${observation.label} · ${observation.remaining_resources} resource units left · recorded P(source bias=0): ${pct(observation.bias_probability_zero)}`
    : 'No observations revealed. The four inherited reports begin the recorded trail.';
  const target = byId('history'); target.replaceChildren();
  for (let index = 0; index < count; index++) {
    const row = element('tr'); const item = history[index];
    if (index === count - 1) row.className = 'current-observation';
    for (const value of [index + 1, `${item.source} / ${item.charged ? 'paid' : 'inherited'}`, item.group, item.label, pct(item.predictive_probability_before_update), item.remaining_resources, pct(item.bias_probability_zero)]) row.append(element('td', String(value)));
    target.append(row);
  }
}
function predictions() {
  if (!current?.result) return;
  const result = current.result;
  const target = byId('predictions'); target.replaceChildren();
  for (let group = 0; group < result.prediction.length; group++) {
    const card = element('div', undefined, 'prediction');
    card.append(element('span', `GROUP ${group}`), element('strong', `${result.initial_prediction[group]} → ${result.prediction[group]}`), element('small', `Initial ${pct(result.initial_confidence[group])} · Final ${pct(result.confidence[group])}`));
    if (truth) card.append(element('small', `Scoring truth: label ${result.truth[group]} · ${result.prediction[group] === result.truth[group] ? 'final correct' : 'final incorrect'}`, 'scoring-truth'));
    target.append(card);
  }
  byId('toggle-truth').textContent = truth ? 'Hide scoring truth' : 'Reveal scoring truth';
  byId('toggle-truth').setAttribute('aria-pressed', String(truth));
  byId('bias').textContent = `Final P(cheap-source bias=0): ${pct(result.bias_probabilities[0])} · paid probes: ${result.probes} · resources used: ${result.resources_used} / 4. Scoring truth is synthetic and was not supplied to the acquisition policy.`;
}
async function benchmark() {
  try {
    const {data} = await api('/api/results');
    if (data.verification?.status !== 'SOURCE_BOUND_AND_RECOMPUTED' || data.verification?.rows_verified !== 5184 || data.data?.primary?.overall_pass !== false) throw new Error('Unqualified benchmark');
    const result = data.data;
    byId('benchmark-message').textContent = '5,184 recorded runs across 576 worlds. The shared-bias gain passed; the clean-sensor cost guard failed. The learned selector did not establish superiority over the strongest simple control.';
    const target = byId('benchmark-tables'); target.replaceChildren();
    target.append(element('p', `Shared-bias utility gain: ${number(result.primary.mean_gain)} · paired 95% interval [${result.primary.ci95.map(number).join(', ')}].`, 'field-note'));
    target.append(element('p', `Clean utility change: ${number(result.clean_guard.mean_gain)} · allowed mean regression: 0.010.`, 'field-note'));
    target.append(element('p', `Learned versus strongest control: ${number(result.strongest_control_comparison.mean_gain)} · paired 95% interval [${result.strongest_control_comparison.ci95.map(number).join(', ')}].`, 'field-note'));
  } catch (_) { byId('benchmark-message').textContent = 'The frozen result is currently unavailable for readback. Its registered overall gate remains FAILED.'; }
}
byId('family').addEventListener('change', descriptions);
byId('policy').addEventListener('change', descriptions);
byId('trial-form').addEventListener('submit', async event => {
  event.preventDefault(); if (running || !ready || retryTimer !== null) return;
  const request = {seed: Number(byId('seed').value), index: Number(byId('episode-index').value), family: byId('family').value, policy: byId('policy').value, model_seed: Number(byId('model-seed').value)};
  running = true; byId('run-trial').disabled = true; byId('trial-message').textContent = 'Running a fresh trial with the verified source and saved model weights…';
  try {
    const result = await api('/api/trial', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(request)});
    render(result.data, result.raw);
    ownTrials.unshift({id: result.data.id, request: result.data.request}); if (ownTrials.length > 12) ownTrials.pop(); sessionList();
  } catch (error) {
    byId('trial-message').textContent = error.retry ? `${error.message} Ready to retry in about ${error.retry} seconds.` : error.message;
    if (error.retry) retryTimer = setTimeout(() => { retryTimer = null; refresh(); }, error.retry * 1000);
  } finally { running = false; await refresh(); }
});
byId('step').addEventListener('input', replay);
byId('refresh-status').addEventListener('click', async () => {
  byId('refresh-status').disabled = true;
  try { await refresh(); await benchmark(); }
  finally { byId('refresh-status').disabled = false; }
});
byId('previous-step').addEventListener('click', () => { byId('step').value = String(Number(byId('step').value) - 1); replay(); });
byId('next-step').addEventListener('click', () => { byId('step').value = String(Number(byId('step').value) + 1); replay(); });
byId('toggle-truth').addEventListener('click', () => { truth = !truth; predictions(); });
byId('export').addEventListener('click', () => {
  if (!current || !currentRaw) return;
  const url = URL.createObjectURL(new Blob([currentRaw], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `szl-confirmation-${current.id}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
descriptions(); refresh(); benchmark();
