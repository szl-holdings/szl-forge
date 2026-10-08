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
  rejectIdentity("unexpected identity field", value => {value.unexpected = "value";});
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
  rejectScore("unexpected response field", value => {value.unexpected = "value";});
  rejectScore("unexpected advisory promotion field", value => {value.advisory.production_promotion_allowed = true;});
  rejectScore("unexpected advisory field", value => {value.advisory.unexpected = "value";});
  for (const key of ["input_sha256", "output_sha256"]) {
    rejectScore("missing " + key, value => {delete value[key];});
    for (const malformed of [null, 0, "", "f".repeat(63), "f".repeat(65), "F".repeat(64), "not-a-hash"]) {
      rejectScore("malformed " + key, value => {value[key] = malformed;});
    }
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

  // The v1 script starts by itself, while v2 makes no request until opt-in.
  let v2Interactions = 0;
  let v2Rejected = 0;
  function checkV2(actual, expected, label) {assert.equal(actual, expected, label); v2Interactions += 1;}
  function rejectV2Identity(label, mutate) {
    const value = clone(fixtures.v2_identity);
    mutate(value);
    assert.throws(() => run("validateV2Identity(candidate)", value), /v2 identity/, label);
    v2Rejected += 1;
  }
  function rejectV2Score(label, mutate) {
    const value = clone(fixtures.v2_healthy);
    mutate(value);
    assert.throws(() => run("validateV2Score(candidate)", value), /v2/, label);
    v2Rejected += 1;
  }
  checkV2(requests.filter(request => request.path.startsWith("/api/v2/")).length, 0, "v2 never auto-enables");
  checkV2(elements.get("v2-score").disabled, true, "v2 score is disabled by default");
  queuedContentType = "application/json";
  queuedResponse = fixtures.identity;
  await elements.get("refresh-identity").listeners.click();
  checkV2(elements.get("score-button").disabled, false, "v1 can recover independently");
  queuedResponse = fixtures.v2_identity;
  await elements.get("v2-enable").listeners.click();
  checkV2(requests.at(-1).path, "/api/v2/identity", "opt-in requests only v2 identity");
  checkV2(requests.at(-1).options.credentials, "omit", "v2 identity omits credentials");
  checkV2(elements.get("v2-status").dataset.state, "ready", "pinned v2 identity is ready");
  checkV2(elements.get("v2-score").disabled, false, "opt-in enables v2 score");
  checkV2(elements.get("score-button").disabled, false, "v1 remains independently available");
  assert.equal(run("validateV2Identity(candidate)", fixtures.v2_identity), fixtures.v2_identity);
  for (const name of ["v2_healthy", "v2_degraded", "v2_ambiguous"]) {
    assert.equal(run("validateV2Score(candidate)", fixtures[name]), fixtures[name].advisory);
  }
  rejectV2Identity("changed Hub revision", value => {value.hub_model.revision = "a".repeat(40);});
  rejectV2Identity("numeric false is not authority false", value => {value.authority.device_control = 0;});
  rejectV2Identity("unexpected clinical field", value => {value.patient_id = "not-allowed";});
  rejectV2Score("wrong prediction set for no alert", value => {value.advisory.prediction_set = [1];});
  rejectV2Score("wrong abstention reason for no alert", value => {value.advisory.abstain_reason = "EMPTY";});
  rejectV2Score("minted receipt", value => {value.receipt_minted = true;});
  rejectV2Score("clinical authority", value => {value.advisory.authority.clinical_decision = true;});
  rejectV2Score("additional advisory field", value => {value.advisory.patient_id = "not-allowed";});
  rejectV2Score("changed embedded Hub revision", value => {value.identity.hub_model.revision = "a".repeat(40);});
  const both = clone(fixtures.v2_ambiguous);
  assert.deepEqual(both.advisory.prediction_set, [0, 1], "backend fixture reaches BOTH");
  checkV2(both.advisory.advisory, "ABSTAIN", "BOTH is an abstention");
  checkV2(run("validateV2Score(candidate)", both).abstain_reason, "BOTH", "UI accepts canonical BOTH abstention");
  // EMPTY is a canonical v2 response shape even if not reached by this authored input.
  const empty = clone(fixtures.v2_healthy);
  empty.advisory.advisory = "ABSTAIN";
  empty.advisory.abstain_reason = "EMPTY";
  empty.advisory.prediction_set = [];
  checkV2(run("validateV2Score(candidate)", empty).abstain_reason, "EMPTY", "UI accepts canonical EMPTY abstention");
  const wrongEmpty = clone(empty);
  wrongEmpty.advisory.prediction_set = [0];
  assert.throws(() => run("validateV2Score(candidate)", wrongEmpty), /v2 advisory/);
  v2Rejected += 1;
  const wrongBoth = clone(both);
  wrongBoth.advisory.prediction_set = [1, 0];
  assert.throws(() => run("validateV2Score(candidate)", wrongBoth), /v2 advisory/);
  v2Rejected += 1;

  await elements.get("preset-healthy").listeners.click();
  queuedResponse = fixtures.v2_healthy;
  await elements.get("v2-score").listeners.click();
  checkV2(requests.at(-1).path, "/api/v2/score", "v2 uses its distinct score route");
  checkV2(requests.at(-1).options.headers["X-SZL-Preview"], "1", "v2 request stays preview-only");
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), {features:fixtures.healthy_features});
  checkV2(elements.get("v2-result").textContent.includes("v2 NO_ALERT"), true, "v2 no-alert rendered");
  checkV2(elements.get("v2-score").disabled, false, "v2 scoring remains available after valid response");
  await elements.get("preset-degraded").listeners.click();
  queuedResponse = fixtures.v2_degraded;
  await elements.get("v2-score").listeners.click();
  assert.deepEqual(JSON.parse(requests.at(-1).options.body), {features:fixtures.degraded_features});
  checkV2(elements.get("v2-result").textContent.includes("v2 ALERT"), true, "v2 alert rendered as advisory only");
  queuedResponse = both;
  await elements.get("v2-score").listeners.click();
  checkV2(elements.get("v2-result").textContent.includes("no determination or action suggested"), true, "BOTH renders no determination");
  queuedResponse = empty;
  await elements.get("v2-score").listeners.click();
  checkV2(elements.get("v2-result").textContent.includes("no determination or action suggested"), true, "EMPTY renders no determination");
  queuedResponse = wrongEmpty;
  await elements.get("v2-score").listeners.click();
  checkV2(elements.get("v2-status").dataset.state, "unavailable", "malformed v2 response refuses closed");
  checkV2(elements.get("v2-score").disabled, true, "malformed v2 response disables v2 scoring");
  checkV2(elements.get("v2-result").textContent.includes("No v2 result displayed"), true, "malformed v2 clears result");
  checkV2(elements.get("score-button").disabled, false, "v2 refusal does not disable v1");
  const afterV2Failure = requests.length;
  await new Promise(setImmediate);
  checkV2(requests.length, afterV2Failure, "v2 refuses without automatic retry");
  queuedResponse = fixtures.v2_identity;
  await elements.get("v2-enable").listeners.click();
  checkV2(elements.get("v2-score").disabled, false, "explicit v2 recheck restores opt-in");
  queuedContentType = "text/html";
  queuedResponse = fixtures.v2_healthy;
  await elements.get("v2-score").listeners.click();
  checkV2(elements.get("v2-score").disabled, true, "non-JSON v2 response disables scoring");
  checkV2(elements.get("score-button").disabled, false, "v1 remains available after v2 transport refusal");
  console.log(JSON.stringify({complete:true, actual_backend_fixtures:2, malformed_contracts_rejected:rejected, interaction_checks:interactions, actual_v2_backend_fixtures:3, v2_malformed_contracts_rejected:v2Rejected, v2_interaction_checks:v2Interactions, scope:"LOCAL_SCRIPT_CONTRACT_ONLY"}));
}
main().catch(error => {console.error(error.stack); process.exitCode = 1;});
