"use strict";

// Execute the actual shipped inline script. This deliberately small DOM adapter
// proves contract/refusal/state behavior, not visual layout or live deployment.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const fixtures = JSON.parse(fs.readFileSync(0, "utf8"));
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1, "one self-contained shipped script is required");
assert(!/\b(?:localStorage|sessionStorage|innerHTML|eval)\b/.test(scripts[0][1]));
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
assert.equal(new Set(ids).size, ids.length, "HTML IDs must be unique");
const elements = new Map();
class Element {
  constructor(id) {
    this.id = id;
    this.textContent = "";
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this.attributes = {};
    this.disabled = false;
    this.checked = false;
    this.value = "";
  }
  get valueAsNumber() {return this.value === "" ? NaN : Number(this.value);}
  append(...children) {this.children.push(...children);}
  replaceChildren(...children) {this.children = [...children];}
  setAttribute(name, value) {this.attributes[name] = value;}
  addEventListener(name, listener) {this.listeners[name] = listener;}
  reportValidity() {return true;}
}
for (const id of ids) elements.set(id, new Element(id));
let queuedResponse = fixtures.identity;
let queuedStatus = 200;
let queuedContentType = "application/json";
let queuedBody = null;
const requests = [];
const sandbox = {
  document: {
    getElementById(id) {assert(elements.has(id), "script uses an existing element: " + id); return elements.get(id);},
    createElement(tag) {return new Element(tag);}
  },
  AbortController,
  TextDecoder,
  setTimeout,
  clearTimeout,
  async fetch(path, options) {
    requests.push({path, options});
    return new Response(queuedBody === null ? JSON.stringify(queuedResponse) : queuedBody, {
      status: queuedStatus,
      headers: {"content-type": queuedContentType}
    });
  }
};
vm.createContext(sandbox);
vm.runInContext(scripts[0][1], sandbox, {filename:"shipped-oac-index.js", timeout:1000});
const clone = (value) => JSON.parse(JSON.stringify(value));
const run = (expression, candidate) => {
  sandbox.candidate = candidate;
  return vm.runInContext(expression, sandbox, {timeout:1000});
};
let rejected = 0;
function rejectIdentity(label, mutate) {
  const value = clone(fixtures.identity);
  mutate(value);
  assert.throws(() => run("validateIdentity(candidate)", value), /identity/, label);
  rejected += 1;
}
function rejectScore(label, mutate) {
  const value = clone(fixtures.healthy);
  mutate(value);
  assert.throws(() => run("validateScore(candidate)", value), /identity|advisory|checks/, label);
  rejected += 1;
}
const authorityNames = ["acknowledgement", "clinical_decision", "device_control", "result_interpretation", "result_release"];

async function main() {
  // Drain the initial asynchronous identity request before interaction tests.
  for (let index = 0; index < 5; index += 1) await new Promise(setImmediate);
  assert.equal(run("serviceReady"), true);
  assert.equal(elements.get("score-button").disabled, false);
  assert.equal(requests.length, 1, "initialization performs only one identity GET");
  assert.equal(requests[0].path, "/api/v1/identity");
  assert.equal(requests[0].options.credentials, "omit");
  assert.equal(requests[0].options.redirect, "error");
  assert.equal(requests[0].options.cache, "no-store");
  assert.equal(run("validateIdentity(candidate)", fixtures.identity), fixtures.identity);
  for (const name of ["healthy", "degraded"]) {
    assert.equal(run("validateScore(candidate)", fixtures[name]), fixtures[name].advisory);
  }
  rejectIdentity("READY requires revision", value => {value.application.revision = null;});
  rejectIdentity("missing revision", value => {delete value.application.revision;});
  rejectIdentity("malformed revision", value => {value.application.revision = "main";});
  rejectIdentity("uppercase revision", value => {value.application.revision = "A".repeat(40);});
  rejectIdentity("UNAVAILABLE identity", value => {value.state = "UNAVAILABLE";});
  rejectIdentity("wrong schema", value => {value.schema = "other";});
  rejectIdentity("wrong application repository", value => {value.application.repository = "other/repo";});
  rejectIdentity("additional application field", value => {value.application.device = true;});
  rejectIdentity("missing application", value => {value.application = null;});
  rejectIdentity("clinical authorization", value => {value.clinical_use_authorized = true;});
  rejectIdentity("production authorization", value => {value.production_promotion_allowed = true;});
  rejectIdentity("missing synthetic scope", value => {delete value.synthetic_training_data;});
  for (const section of ["artifact_source", "hub_model", "hub_dataset", "artifacts"]) {
    for (const key of Object.keys(fixtures.identity[section])) {
      rejectIdentity(section + "." + key + " must match pin", value => {value[section][key] = "c".repeat(64);});
    }
    rejectIdentity("additional " + section + " field", value => {value[section].unexpected = "value";});
    rejectIdentity("missing " + section, value => {delete value[section];});
  }
  rejectScore("not ok", value => {value.ok = false;});
  rejectScore("minted receipt", value => {value.receipt_minted = true;});
  rejectScore("missing receipt flag", value => {delete value.receipt_minted;});
  rejectScore("missing advisory", value => {value.advisory = null;});
  rejectScore("score identity null binding", value => {value.identity.application.revision = null;});
  rejectScore("wrong advisory schema", value => {value.advisory.schema = "other";});
  rejectScore("wrong semantics", value => {value.advisory.score_semantics = "clinical_probability";});
  rejectScore("wrong purpose", value => {value.advisory.purpose = "diagnosis";});
  for (const value of [-0.1, 1.1, NaN, Infinity, "0.5", null]) {
    rejectScore("invalid score " + String(value), response => {response.advisory.operator_attention_score = value;});
  }
  for (const value of [-0.1, 1.1, NaN, Infinity, "0.5"]) {
    rejectScore("invalid threshold " + String(value), response => {response.advisory.decision_threshold = value;});
  }
  rejectScore("inconsistent attention decision", value => {value.advisory.operator_attention_required = !value.advisory.operator_attention_required;});
  rejectScore("non-boolean attention decision", value => {value.advisory.operator_attention_required = 0;});
  for (const name of authorityNames) rejectScore("authority must remain false: " + name, value => {value.advisory.authority[name] = true;});
  rejectScore("missing authority", value => {delete value.advisory.authority.device_control;});
  rejectScore("extra authority", value => {value.advisory.authority.production = false;});
  rejectScore("nonfinite contribution", value => {value.advisory.normalized_feature_contributions.queue_utilization = NaN;});
  rejectScore("missing contribution", value => {delete value.advisory.normalized_feature_contributions.queue_utilization;});
  rejectScore("extra contribution", value => {value.advisory.normalized_feature_contributions.patient = 0;});

  let interactions = 0;
  function check(actual, expected, label) {assert.equal(actual, expected, label); interactions += 1;}
  for (const name of ["healthy", "degraded"]) {
    queuedResponse = fixtures[name];
    await elements.get("preset-" + name).listeners.click();
    check(elements.get("score-value").textContent, "—", "preset clears old score");
    await elements.get("score-form").listeners.submit({preventDefault() {}});
    check(elements.get("score-value").textContent, fixtures[name].advisory.operator_attention_score.toFixed(3), "renders actual backend score");
    check(elements.get("contributions").children.length, 8, "renders exactly eight contributions");
    check(elements.get("advisory-panel").attributes["aria-busy"], "false", "busy state cleared");
    check(elements.get("score-button").disabled, false, "controls re-enabled");
    const request = requests.at(-1);
    check(request.path, "/api/score", "only canonical score route");
    check(request.options.headers["X-SZL-Preview"], "1", "preview header supplied");
    assert.deepEqual(JSON.parse(request.options.body), {features: fixtures[name + "_features"]});
    interactions += 1;
  }
  elements.get("score-form").listeners.input();
  check(elements.get("score-value").textContent, "—", "input change clears stale score");
  check(elements.get("contributions").children.length, 1, "input change clears contributions");
  queuedResponse = clone(fixtures.healthy);
  queuedResponse.identity.application.revision = null;
  await elements.get("score-form").listeners.submit({preventDefault() {}});
  check(elements.get("score-value").textContent, "—", "malformed response never keeps a score");
  check(elements.get("score-button").disabled, true, "refusal requires explicit identity refresh");
  check(elements.get("service-status").dataset.state, "unavailable", "refusal sets unavailable");
  const afterFailure = requests.length;
  await new Promise(setImmediate);
  check(requests.length, afterFailure, "no retry after contract refusal");
  queuedResponse = fixtures.identity;
  await elements.get("refresh-identity").listeners.click();
  check(elements.get("score-button").disabled, false, "explicit refresh restores ready state");
  check(elements.get("score-value").textContent, "—", "identity refresh does not invent score");
  queuedBody = "x".repeat(65537);
  await elements.get("score-form").listeners.submit({preventDefault() {}});
  check(elements.get("score-value").textContent, "—", "oversized response never displays score");
  check(elements.get("score-button").disabled, true, "oversized response disables scoring");
  queuedBody = null;
  queuedResponse = fixtures.identity;
  await elements.get("refresh-identity").listeners.click();
  queuedContentType = "text/html";
  queuedResponse = fixtures.healthy;
  await elements.get("score-form").listeners.submit({preventDefault() {}});
  check(elements.get("score-value").textContent, "—", "non-JSON response cannot display score");
  check(elements.get("score-button").disabled, true, "non-JSON response disables scoring");
  console.log(JSON.stringify({complete:true, actual_backend_fixtures:2, malformed_contracts_rejected:rejected, interaction_checks:interactions, scope:"LOCAL_SCRIPT_CONTRACT_ONLY"}));
}
main().catch(error => {console.error(error.stack); process.exitCode = 1;});
