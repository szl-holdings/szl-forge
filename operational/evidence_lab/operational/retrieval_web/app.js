"use strict";
const el = (id) => document.getElementById(id);
let mode = "corpus", ready = false, busy = false, summary = null;
const percentage = (value) => `${(value * 100).toFixed(1)}%`;
function executionDeviceLabel() {
  const device = String(summary?.execution_device || "").toLowerCase();
  if (device === "cpu") return "CPU";
  if (/^cuda(?::\d+)?$/.test(device)) return "GPU";
  return "Runtime";
}
function controls() {
  ["submit", "search-only", "example-positive", "example-negative", "corpus-tab", "context-tab"].forEach(id => el(id).disabled = busy || !ready);
  el("question").disabled = busy; el("context").disabled = busy;
  el("identifier-guard").disabled = busy || !ready || !summary?.capabilities?.identifier_guard;
  el("question-form").setAttribute("aria-busy", String(busy));
}
function switchMode(next) {
  if (busy) return;
  mode = next;
  for (const [id, value] of [["corpus-tab", "corpus"], ["context-tab", "context"]]) {
    el(id).classList.toggle("selected", mode === value); el(id).setAttribute("aria-selected", String(mode === value)); el(id).tabIndex = mode === value ? 0 : -1;
  }
  el("question-panel").setAttribute("aria-labelledby", mode === "corpus" ? "corpus-tab" : "context-tab");
  el("context-fields").hidden = mode !== "context"; el("context").required = mode === "context";
  el("search-only").hidden = mode !== "corpus";
  el("guard-fields").hidden = mode !== "corpus";
  el("submit-label").textContent = mode === "corpus" ? "Retrieve & answer" : "Check this passage";
}
function message(text, kind = "") { el("request-status").textContent = text; el("request-status").className = `request-status ${kind}`; }
function questionCount() { el("question-count").textContent = `${Array.from(el("question").value).length} / 512`; }
function addText(parent, tag, text, className) { const node = document.createElement(tag); node.textContent = text; if (className) node.className = className; parent.append(node); return node; }
function safeSource(url) {
  try { const parsed = new URL(url); if (parsed.protocol === "https:" && ["rajpurkar.github.io", "hotpotqa.github.io", "huggingface.co", "aclanthology.org", "creativecommons.org"].includes(parsed.hostname)) return parsed.href; } catch (_) {}
  return null;
}
function renderPassages(data) {
  el("passages").replaceChildren();
  const passages = data.passages || [];
  el("passage-count").textContent = `${passages.length} ${passages.length === 1 ? "passage" : "passages"}`;
  passages.forEach((doc, index) => {
    const card = document.createElement("article"); card.className = "passage";
    const title = document.createElement("h4"); addText(title, "span", String(index + 1).padStart(2, "0")); title.append(document.createTextNode(doc.title.replaceAll("_", " "))); card.append(title);
    const p = document.createElement("p"), points = Array.from(doc.text), evidence = data.evidence;
    if (evidence && evidence.document_id === doc.id && Number.isInteger(evidence.start) && Number.isInteger(evidence.end) && evidence.start >= 0 && evidence.end <= points.length && points.slice(evidence.start, evidence.end).join("") === data.answer) {
      p.append(document.createTextNode(points.slice(0, evidence.start).join(""))); addText(p, "mark", data.answer); p.append(document.createTextNode(points.slice(evidence.end).join("")));
    } else { p.textContent = doc.text; }
    card.append(p);
    const citation = document.createElement("div"); citation.className = "citation";
    if (doc.citation && typeof doc.citation === "object") {
      const url = safeSource(doc.citation.publisher_url);
      if (url) { const link = addText(citation, "a", `${doc.citation.dataset_id} ↗`); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; }
      else addText(citation, "span", doc.citation.dataset_id || "Dataset source");
      addText(citation, "span", ` · ${doc.citation.license} · ${doc.citation.attribution}`);
    } else { citation.textContent = String(doc.citation || "Source not supplied"); }
    card.append(citation); el("passages").append(card);
  });
}
function render(data) {
  el("empty-result").hidden = true; el("answer-result").hidden = false;
  el("answer-status").className = `answer-status ${data.status === "ABSTAIN" ? "abstain" : ""}`;
  el("answer-status").textContent = {ANSWER: "Extracted from a source", ABSTAIN: "Reader abstained", RETRIEVED: "Passages retrieved"}[data.status] || data.status;
  el("answer-text").textContent = data.status === "ANSWER" ? data.answer : data.status === "ABSTAIN" ? "No answer accepted from this evidence." : "Inspect the relevant passages.";
  el("answer-scope").textContent = data.scope === "provided_context_only"
    ? "This decision concerns only your supplied passage. The reader’s calibration may not transfer to other kinds of text."
    : data.status === "RETRIEVED" ? "Retrieval ranks passages; it does not verify an answer. Inspect the source text before drawing a conclusion."
    : "This decision concerns the three retrieved passages only. An abstention does not prove that no answer exists elsewhere. Corpus-wide false-answer risk has not been measured.";
  el("elapsed").textContent = `${data.elapsed_seconds.toFixed(2)} s · local inference`;
  if (data.identifier_guard?.reason === "UNRESOLVED_IDENTIFIER") {
    el("answer-status").textContent = "Identifier guard abstained";
    el("answer-scope").textContent = "The question includes an identifier-like token absent from the frozen corpus. This experimental spelling check does not establish semantic answerability.";
  }
  el("margin-row").hidden = typeof data.margin !== "number";
  if (typeof data.margin === "number") el("margin-row").textContent = `Raw margin ${data.margin.toFixed(2)} / threshold ${data.threshold.toFixed(2)} · not a confidence probability`;
  renderPassages(data);
  const {passages, ...receipt} = data; el("response-receipt").textContent = JSON.stringify(receipt, null, 2);
}
async function submit(searchOnly = false) {
  if (!ready || busy || !el("question-form").reportValidity()) return;
  const question = el("question").value.trim();
  if (!question) { message("Enter a question first.", "error"); return; }
  busy = true; controls(); message(searchOnly ? "Retrieving passages from the local index…" : "Reading the evidence locally…", "busy");
  el("answer-result").hidden = true; el("empty-result").hidden = true;
  const endpoint = mode === "context" ? "answer-context" : searchOnly ? "search" : el("identifier-guard").checked ? "query-guarded" : "query";
  const body = {question}; if (mode === "context") body.context = el("context").value; else if (searchOnly) body.k = 5;
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(`/api/${endpoint}`, {method: "POST", headers: {"Content-Type": "application/json", "X-SZL-Preview": "1"}, body: JSON.stringify(body), signal: controller.signal, cache: "no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    render(data); message("Complete. Inspect the source passages and evidence scope.");
  } catch (error) {
    el("empty-result").hidden = false;
    message(error.name === "AbortError" ? "The request timed out. Local inference may still be finishing; wait before retrying." : error.message || "The local service is unavailable.", "error");
  } finally { clearTimeout(timer); busy = false; controls(); }
}
async function initialize() {
  controls();
  try {
    const response = await fetch("/api/status", {cache: "no-store", signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error("Runtime status unavailable. Restart the local preview and reload this page.");
    summary = await response.json();
    if (summary.status !== "LOCAL_READY_NOT_PRODUCTION") throw new Error("The local models are not ready.");
    el("health").textContent = `${executionDeviceLabel()} ready · local`; el("document-count").textContent = summary.documents.toLocaleString();
    el("metric-retrieval").textContent = percentage(summary.metrics.squad_retrieval.hybrid_rrf.recall_at_10);
    el("metric-multihop").textContent = percentage(summary.metrics.hotpot_support_retrieval.qwen.all_support_at_5);
    const answerability = summary.metrics.given_context_answerability;
    el("metric-false").textContent = `${answerability.false_answer_count} / ${answerability.unanswerable_count}`;
    el("record-limits").textContent = `Given-context false-answer rate: ${percentage(answerability.false_answer_rate)} (95% interval ${percentage(answerability.false_answer_rate_wilson_ci[0])}–${percentage(answerability.false_answer_rate_wilson_ci[1])}). Retrieved answering: ${percentage(summary.metrics.retrieved_context_qa_answerable_only.exact_match)} exact match on answerable queries. Public development sets; model contamination is unknown. These are not corpus-wide unsupported-query guarantees.`;
    const fields = {"Encoder": summary.models.encoder, "Reader": summary.models.reader, "Execution device": summary.execution_device || "Unrecorded", "Recorded GPU": summary.device || "None", "Run": summary.run_id, "Evaluation completed": summary.completed_at, "Source fingerprint": summary.freeze_sha256, "Result fingerprint": summary.result_sha256};
    for (const [label, value] of Object.entries(fields)) { addText(el("provenance"), "dt", label); addText(el("provenance"), "dd", String(value)); }
    ready = true; controls(); message("Ready. Choose an example or ask your own question.");
  } catch (error) { el("health").textContent = "Runtime unavailable"; el("health").classList.add("error"); message(error.message, "error"); }
}
el("corpus-tab").addEventListener("click", () => switchMode("corpus")); el("context-tab").addEventListener("click", () => switchMode("context"));
for (const id of ["corpus-tab", "context-tab"]) el(id).addEventListener("keydown", (event) => { if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key) && !busy) { event.preventDefault(); switchMode(event.key === "Home" ? "corpus" : event.key === "End" ? "context" : mode === "corpus" ? "context" : "corpus"); el(mode === "corpus" ? "corpus-tab" : "context-tab").focus(); } });
el("question").addEventListener("input", questionCount);
el("question-form").addEventListener("submit", event => { event.preventDefault(); submit(); }); el("search-only").addEventListener("click", () => submit(true));
el("example-positive").addEventListener("click", () => { const example = summary.examples[0]; el("question").value = example.question; el("context").value = example.context; questionCount(); el("question").focus(); });
el("example-negative").addEventListener("click", () => { const example = summary.examples[1]; switchMode("context"); el("question").value = example.question; el("context").value = example.context; questionCount(); el("question").focus(); });
initialize();
