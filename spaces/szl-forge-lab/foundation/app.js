'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const labels = {
    clean: 'Clean observations', shared_bias: 'Shared source bias',
    independent_outliers: 'Independent cheap outliers', reference_outliers: 'Reference outliers',
    bias_drift: 'Bias changes after acquisition', compositional: 'Compositional changes',
    no_confirmation: 'No extra confirmation', independent_bias: 'Independent-bias system',
    source_aware: 'Source-aware analytic', cheap_only: 'Paid cheap only',
    reference_only: 'Paid reference only', random_mixed: 'Random mixed', learned: 'Learned selector'
  };
  const descriptions = {
    clean: 'Both channels follow the stated fresh-noise likelihood. Extra confirmation can waste resources.',
    shared_bias: 'The cheap source applies one hidden offset across reports. Repetition preserves that dependence.',
    independent_outliers: 'Cheap observations have additional independent corruption beyond the inference model.',
    reference_outliers: 'The reference source is independently corrupted beyond the assumed likelihood.',
    bias_drift: 'The source bias changes after the first paid cheap observation. This condition depends on the action history.',
    compositional: 'Two target groups change together outside the supplied structured world families.'
  };
  const policyDescriptions = {
    no_confirmation: 'Uses the four inherited cheap reports and buys no further evidence.',
    independent_bias: 'Models the per-reading bias distribution as independent across observations.',
    source_aware: 'Uses the explicit shared-bias posterior and greedy one-step expected acquisition utility.',
    cheap_only: 'Uses shared-bias inference and restricts additional paid acquisition to the cheap channel.',
    reference_only: 'Uses shared-bias inference and restricts additional paid acquisition to the reference channel.',
    random_mixed: 'Chooses uniformly among affordable actions until resources are exhausted.',
    learned: 'A trained 26,792-parameter selector chooses from raw public history; the Bayesian predictor remains programmed.'
  };
  const state = {catalog:null, summary:null, rows:[], cache:new Map(), request:0, truth:false};
  const controls = ['episode','previous-episode','next-episode','model-seed','policy','price','reset-price','step','previous-step','next-step','toggle-truth','export-row'];
  const num = value => Number(value).toFixed(3);
  const signed = value => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;
  const pct = value => `${(Number(value)*100).toFixed(1)}%`;
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  };
  function enabled(value) { controls.forEach(id => { $(id).disabled = !value; }); }
  function fail(error) {
    enabled(false); $('family').disabled = true;
    $('load-status').textContent = `Evidence unavailable: ${error.message}. Download the frozen release to inspect it locally.`;
    $('load-status').classList.add('error-banner');
    $('episode-status').textContent = 'Replay disabled because evidence could not be admitted.';
  }
  async function fetchJSON(path, expected) {
    const response = await fetch(path, {cache:'no-cache'});
    if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (expected) {
      if (!globalThis.crypto?.subtle) throw new Error('This explorer requires HTTPS or localhost for evidence hashing');
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2,'0')).join('');
      if (digest !== expected) throw new Error(`Evidence hash mismatch: ${path}`);
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  function fillSelect(id, values, format = value => labels[value] || String(value)) {
    $(id).replaceChildren(...values.map(value => {
      const option = node('option', format(value)); option.value = String(value); return option;
    }));
  }
  function definitionList(id, values) {
    $(id).replaceChildren(...Object.entries(values).flatMap(([key,value]) => [node('dt',key),node('dd',value)]));
  }
  function renderFindings() {
    const s = state.summary, primary = s.primary, clean = s.clean_guard, strongest = s.strongest_control_comparison;
    $('overall-result').textContent = primary.overall_pass ? 'PASSED' : 'FAILED';
    $('gain-value').textContent = signed(primary.mean_gain);
    $('gain-status').textContent = primary.gain_gate_pass ? 'GAIN CRITERION PASSED' : 'GAIN CRITERION FAILED';
    $('gain-status').classList.add(primary.gain_gate_pass ? 'pass' : 'fail');
    $('gain-ci').textContent = `95% interval [${signed(primary.ci95[0])}, ${signed(primary.ci95[1])}]. Learned minus independent-bias utility.`;
    $('clean-value').textContent = signed(clean.mean_gain);
    $('clean-status').textContent = primary.clean_guard_pass ? 'CLEAN GUARD PASSED' : 'CLEAN GUARD FAILED';
    $('clean-status').classList.add(primary.clean_guard_pass ? 'pass' : 'fail');
    $('clean-ci').textContent = `95% interval [${signed(clean.ci95[0])}, ${signed(clean.ci95[1])}]. Allowed mean regression: 0.010.`;
    $('strongest-result').textContent = `The learned selector did not outperform the strongest control (${labels[strongest.observed_best_control]}). Its utility difference was ${signed(strongest.mean_gain)}, with a 95% interval of [${signed(strongest.ci95[0])}, ${signed(strongest.ci95[1])}].`;
    definitionList('release-provenance', {
      'Research release': state.catalog.release_name,
      'Evidence': `${state.catalog.row_count.toLocaleString()} retained rollouts; 576 paired synthetic worlds`,
      'Archive SHA-256': state.catalog.archive_sha256,
      'Protocol SHA-256': state.catalog.protocol_sha256,
      'Evaluation journal SHA-256': state.catalog.evaluation_sha256,
      'Summary SHA-256': state.catalog.summary_sha256,
      'Publication binding': state.catalog.source_revision_note,
      'Browser execution': 'Recorded evidence replay; no model inference or training'
    });
  }
  function episodeIndex() { return Math.max(0, Math.min(95, Math.trunc(Number($('episode').value) || 0))); }
  function rowFor(policy) {
    const seed = policy === 'learned' ? Number($('model-seed').value) : null;
    const row = state.rows.find(r => r.episode_index === episodeIndex() && r.policy === policy && r.model_seed === seed);
    if (!row) throw new Error(`Missing recorded row for ${policy}, episode ${episodeIndex()}`);
    return row;
  }
  function score(row) { return row.accuracy - row.paid_cost * Number($('price').value); }
  function renderScores() {
    const price = Number($('price').value);
    $('price-output').textContent = `${price.toFixed(2)}×`;
    $('utility-heading').textContent = price === 1 ? 'Utility' : 'Rescored utility';
    $('episode-table').replaceChildren(...state.catalog.policies.map(policy => {
      const r = rowFor(policy), tr = node('tr');
      if (policy === $('policy').value) tr.classList.add('selected');
      tr.append(node('th',labels[policy]),node('td',pct(r.accuracy)),node('td',num(r.paid_cost)),node('td',num(score(r))));
      tr.firstChild.scope = 'row'; return tr;
    }));
    const metrics = state.summary.summaries[$('family').value];
    const values = state.catalog.policies.map(p => metrics[p].accuracy - metrics[p].paid_cost * price);
    const low = Math.min(0,...values), high = Math.max(1,...values);
    $('aggregate-bars').replaceChildren(...state.catalog.policies.map((policy,i) => {
      const container = node('div',undefined,'aggregate-row');
      const label = node('div',undefined,'bar-label'); label.append(node('span',labels[policy]),node('strong',num(values[i])));
      const track = node('div',undefined,'bar-track'), fill = node('span',undefined,'bar-fill');
      fill.style.display = 'block'; fill.style.width = `${(values[i]-low)/(high-low)*100}%`; track.append(fill);
      container.append(label,track,node('p',`Accuracy ${pct(metrics[policy].accuracy)} · paid cost ${num(metrics[policy].paid_cost)} · confident error ${pct(metrics[policy].confident_wrong)}`,'small'));
      return container;
    }));
    $('aggregate-bars').append(node('p',`Bar scale: ${num(low)} to ${num(high)} utility. ${price === 1 ? 'Registered observation prices.' : 'Post-hoc prices; recorded actions stay fixed.'}`,'small'));
  }
  function renderReplay(resetStep = false) {
    const row = rowFor($('policy').value);
    $('step').max = row.history.length;
    if (resetStep) $('step').value = row.history.length;
    const step = Math.max(0,Math.min(row.history.length,Number($('step').value)));
    $('step-output').textContent = `${step} / ${row.history.length}`;
    const last = step ? row.history[step-1] : null;
    $('current-source').textContent = last ? `${last.source === 'reference' ? 'Reference' : 'Cheap'} · group ${last.group+1}` : 'Before replay';
    $('current-cost').textContent = last ? num(last.paid_cost_total) : '0.000';
    $('current-budget').textContent = last ? last.remaining_resources : row.public.budget;
    $('current-bias').textContent = last ? pct(last.bias_probability_zero) : 'Not revealed';
    $('policy-description').textContent = policyDescriptions[row.policy];
    $('observation-trail').replaceChildren(...row.history.map((observation,i) => {
      const card = node('li',undefined,'observation-card');
      if (i >= step) { card.append(node('span',`Observation ${i+1}`,'observation-label'),node('strong','Not revealed')); return card; }
      card.classList.add('revealed');
      card.append(node('span',`${i+1} / ${observation.charged ? 'PAID' : 'INHERITED'}`,'observation-label'),node('strong',`${observation.source} → ${observation.label}`),node('p',`Target group ${observation.group+1} · total paid ${num(observation.paid_cost_total)}`,'observation-detail'));
      return card;
    }));
    $('prediction-grid').replaceChildren(...row.prediction.map((prediction,i) => {
      const card = node('div',undefined,'prediction-card');
      card.append(node('span',`GROUP ${i+1} · weight ${pct(row.public.goal_mass[i])}`,'observation-label'));
      const values = node('div',undefined,'prediction-values');
      values.append(node('p',`Initial ${row.initial_prediction[i]} (${pct(row.initial_confidence[i])})`),node('p',`Final ${prediction} (${pct(row.confidence[i])})`)); card.append(values);
      card.append(node('div',state.truth ? `Scoring truth: ${row.truth[i]}` : 'Scoring truth hidden','prediction-truth')); return card;
    }));
    $('toggle-truth').setAttribute('aria-pressed',String(state.truth));
    $('toggle-truth').textContent = state.truth ? 'Hide scoring truth' : 'Reveal scoring truth';
    definitionList('row-provenance', {
      'World': `${row.family} / episode ${row.episode_index} / generator seed ${row.seed}`,
      'Policy': `${labels[row.policy]}${row.model_seed === null ? '' : ` / checkpoint seed ${row.model_seed}`}`,
      'Initial predictions': 'Retained endpoint after all four inherited reports, regardless of replay step',
      'Final predictions': 'Retained endpoint after all recorded actions, regardless of replay step',
      'Row SHA-256': row.row_sha256, 'Protocol SHA-256': row.protocol_sha256,
      'Family file SHA-256': state.catalog.episodes[row.family].sha256
    });
  }
  function render(resetStep = false) {
    if (!state.rows.length) return;
    $('episode').value = episodeIndex();
    renderScores(); renderReplay(resetStep);
    $('episode-status').textContent = `Replay: ${labels[$('family').value]}, episode ${episodeIndex()}. Seven policies share this world. Evidence file hash matches the packaged catalog.`;
  }
  async function loadFamily() {
    const requested = ++state.request, family = $('family').value;
    enabled(false); $('episode-status').textContent = 'Loading and hashing recorded evidence…';
    try {
      let rows = state.cache.get(family);
      if (!rows) {
        const binding = state.catalog.episodes[family];
        rows = await fetchJSON(binding.path,binding.sha256);
        if (!Array.isArray(rows) || rows.length !== binding.row_count || rows.some(r => r.family !== family || r.protocol_sha256 !== state.catalog.protocol_sha256)) throw new Error('Family evidence population mismatch');
        state.cache.set(family,rows);
      }
      if (requested !== state.request) return;
      state.rows = rows; $('family-description').textContent = descriptions[family];
      enabled(true); render(true);
    } catch(error) { if (requested === state.request) fail(error); }
  }
  function safeRender(reset = false) { try {render(reset);} catch(error) {fail(error);} }
  $('family').addEventListener('change',loadFamily);
  $('episode').addEventListener('change',() => safeRender(true));
  $('previous-episode').addEventListener('click',() => { $('episode').value = Math.max(0,episodeIndex()-1); safeRender(true); });
  $('next-episode').addEventListener('click',() => { $('episode').value = Math.min(95,episodeIndex()+1); safeRender(true); });
  ['policy','model-seed'].forEach(id => $(id).addEventListener('change',() => safeRender(true)));
  $('price').addEventListener('input',() => safeRender());
  $('reset-price').addEventListener('click',() => { $('price').value = 1; safeRender(); });
  $('step').addEventListener('input',() => safeRender());
  $('previous-step').addEventListener('click',() => { $('step').value = Math.max(0,Number($('step').value)-1); safeRender(); });
  $('next-step').addEventListener('click',() => { $('step').value = Math.min(Number($('step').max),Number($('step').value)+1); safeRender(); });
  $('toggle-truth').addEventListener('click',() => {state.truth = !state.truth; safeRender();});
  $('export-row').addEventListener('click',() => {
    const row = rowFor($('policy').value);
    const url = URL.createObjectURL(new Blob([JSON.stringify(row,null,2)+'\n'],{type:'application/json'}));
    const link = node('a'); link.href = url; link.download = `${row.family}-${row.episode_index}-${row.policy}${row.model_seed === null ? '' : '-'+row.model_seed}.json`;
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
  });
  async function start() {
    enabled(false);
    state.catalog = await fetchJSON('data/catalog.json');
    const c = state.catalog;
    if (c.schema !== 'szl.confirmation.showcase/v1' || c.row_count !== 5184 || c.archive_sha256 !== '869e318dd5f328205dd181ee836ef267bd2ae278f6430a9e8ddc661fbc689d03' || c.summary_sha256 !== '3db93276b4a26b325aab7a33254e1376a4da607cfb0daa46dec150b770f597fd') throw new Error('Unexpected frozen release catalog');
    state.summary = await fetchJSON('data/summary.json',c.summary_sha256);
    if (state.summary.primary.overall_pass !== false) throw new Error('Frozen gate outcome mismatch');
    fillSelect('family',c.families); fillSelect('policy',c.policies); fillSelect('model-seed',c.model_seeds,value => `Checkpoint ${value}`);
    $('family').value = 'shared_bias'; $('policy').value = 'learned'; $('family').disabled = false;
    renderFindings();
    $('load-status').textContent = 'Frozen summary hash verified. Explore all 5,184 retained records; each environment is hashed before replay. No new inference runs here.';
    await loadFamily();
  }
  start().catch(fail);
})();
