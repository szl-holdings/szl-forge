# SZL Access Lab

A local-first accessibility repair pilot: submit a small synthetic HTML form,
inspect narrowly scoped missing label associations, propose a repair, and retain
an operational receipt describing the gate result. The frontend displays source
as text; submitted HTML is not executed.

This is a working application and deterministic baseline, **not newly trained
model weights, WCAG certification, or a claim of complete accessibility**.
The initial repair is deliberately small: associate a plain adjacent `label`
with an existing uniquely identified `input` without changing the label's words,
the control, or application behavior. Complex, ambiguous, dynamic, hidden, and
out-of-scope markup is left for review.

## Install and start

Use Python 3.11 through 3.13. From the repository root, create an isolated environment
and install the package:

```powershell
python -m venv .venv-access
.\.venv-access\Scripts\Activate.ps1
python -m pip install ./access-lab
```

Generate a session key and choose a **new trusted local directory** for durable
state. Do not use a shared folder, a public web directory, or a patient-data folder.

```powershell
$env:SZL_ACCESS_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
szl-access --state-dir .\access-state-pilot --port 8030
```

Open `http://127.0.0.1:8030/` and enter the same session key in the UI. The API
requires a key of at least 32 characters. Retain it privately; do not commit it,
paste it into an issue, or include it in a screenshot. Use the bundled synthetic
examples for the first run.

The service binds to loopback. This pilot is not configured for public hosting,
multi-user tenancy, remote device control, or medical result handling. Local
authentication is not a substitute for a hardened operating-system account.

## Optional existing local model

The deterministic baseline needs no model. To enable the optional model proposal
path, set an **already installed** Ollama model before starting the server:

```powershell
$env:SZL_ACCESS_OLLAMA_MODEL = "<exact-existing-local-model-name>"
szl-access --state-dir .\access-state-model-pilot --port 8030
```

Only the local Ollama service at `127.0.0.1:11434` is contacted, with proxies and
redirects disabled. **A loopback address alone does not prove local execution**:
Ollama can also expose cloud-backed models. Before sending candidate text, this
adapter rejects cloud-style names and remote metadata and requires an installed
GGUF tag digest, architecture details, and an absolute local SHA-256 blob
reference. It checks the reported tag identity before and after generation.
Missing, ambiguous, remote, changed, or unsupported metadata prevents acceptance.

These are checks of a **trusted local daemon's reports**, not independent hashing
of the loaded weights or verification that the server's cloud features are
disabled. The receipt explicitly labels server-wide cloud-disable state as
`NOT_VERIFIED`. Keep cloud execution disabled in your owned Ollama configuration;
this application does not change, restart, or harden that separate server.

This application does not pull models, start cloud jobs, or train weights. Its
planned cloud-compute spend is zero; existing hardware still consumes electricity,
storage, and time. Every proposal passes the same deterministic patch gate. A
model's own claim of success is not accepted as verification, and an unavailable
model is reported as unavailable rather than silently relabeled as a successful
model evaluation.

## What the gate proves

The accepted input grammar requires well-formed restricted static markup, quoted
attribute values, unique simple IDs, and unambiguous adjacency within one parent.
An accepted patch only adds the matching `for` attribute to the original label.
It does not rewrite a whole page or guess a new label from personal information.

| State | Meaning |
| --- | --- |
| `VERIFIED_SCOPED_REPAIR` | The allowed label-binding patch passed this restricted gate. |
| `NO_CHANGE` | No allowed repair was applied. This does not mean the page is fully accessible. |
| `REVIEW_REQUIRED` | The source or proposal needs review; do not treat the output as repaired. |

Source and output hashes bind the receipt to exact bytes. Local durable records
are operational receipts, not independently trusted signatures or an external
attestation that a user with a disability successfully completed a task.

The **Download source-free summary** button exports an authenticated, allowlisted
JSON summary: state, method, timestamp, run identifier, and source/output/receipt
hashes. It excludes HTML, binding text, and model details. The summary's
`export_sha256` covers its canonical JSON fields except that hash; its
`receipt_sha256` refers to the separately retained full local receipt. This is
not an anonymization guarantee: hashes and run metadata can identify a known
document. Review exports before sharing, and retain full local records for an
actual repair review. The same summary is available to authenticated local
clients at `GET /api/runs/<run-id>/summary`.

Stored records are size-bounded and checked for unique JSON fields, identity,
recognized state, timestamp, receipt digest, and output-byte digest. Corrupt
records fail closed with `LOCAL_RECEIPT_INTEGRITY_FAILED`; they are not silently
rewritten or returned as successful results. Startup also validates retained
records before recovering interrupted work. Pre-release unfinished `RUNNING`
records without a receipt digest are not automatically upgraded: preserve that
old directory for inspection and start a new explicitly chosen pilot directory.
Previously completed valid receipts remain readable. These checks detect damage
and inconsistent records, not deliberate rewriting by an owner who can replace
both data and hashes.

## Reproduce the public synthetic baseline

```powershell
python -m szl_access.benchmark
python -m pip install pytest
python -m pytest access-lab/tests -q
```

The benchmark prints JSON and exits nonzero if any expected state or integrity
check fails. Its initial 36 SZL-authored synthetic cases cover 12 scoped repairs,
18 refusals, and six no-change cases. They include distinct forms, existing
bindings, duplicate IDs, malformed markup, hidden/ARIA semantics, and active
content. The report contains a recipe hash, per-case source/output hashes,
family/split-group labels, failed IDs, and state counts. Every output must match
an exact authored fixture expectation, and successful repairs must also be
idempotent. Consistent hashes alone do not make an unintended edit pass.

These are **public development regression fixtures**: they are visible in source
and used while building the engine. A perfect score is not an unseen evaluation,
generalization estimate, browser/assistive-technology result, or model-training
claim. Do not train on this suite and then present its score as held-out accuracy.
For later paired proposer comparisons, `run_benchmark(proposer,
methodology="candidate-name")` feeds a caller-provided proposer through the same
gate; the name is not independent proof of the executed model's identity.

Before a specialist fine-tune, collect separately licensed/owned cases, deduplicate
by component/template family, reserve an unseen family-separated test set, pin
the parent/candidate artifacts and prompt, and compare accepted repairs and
regressions against the deterministic baseline. Training is justified only if
the candidate adds measured capability. No new model repository or weights are
required to run this pilot.

The [installed-application CI](../.github/workflows/szl-access-lab.yml) builds a
wheel and installs it into a fresh environment across Python 3.11–3.13 on Linux
and Windows. Tests run with isolated Python from outside the source checkout.
The [installed verifier](tools/verify_installed.py) checks package origin, exact
wheel/module/UI bytes, CSP hashes, authentication, scoped repair, durable receipt
replay and restart, and shutdown/temporary-state cleanup. It uses synthetic input
and disables model invocation. This is an installed HTTP/resource contract test,
not a browser or assistive-technology usability test; those need separate evidence.

## Privacy and retention

Start with the synthetic examples. Submitted source and repair records persist
in the chosen local state directory for restart/replay workflows; this is not a
zero-retention service. Do not submit passwords, access tokens, personal records,
private customer pages, patient data, or third-party material you are not allowed
to process. Optional local inference sends source-related context to the locally
configured Ollama process; its own logs and model-server configuration are outside
this application's storage boundary.

There is no scraping, arbitrary URL fetching, HTML execution, cloud telemetry, or
automatic publication workflow in this pilot. Review retained files before
sharing a receipt or backup. To retire a disposable test workspace, stop the
server and remove only that explicitly identified state directory; keep required
records according to your own retention policy.

## Product hypothesis, not a revenue claim

The proposed first customers are small web agencies and nonprofits maintaining
forms. Paid setup, scoped remediation support, and repeatable regression checks
are hypotheses to validate with consenting pilot users. There are no measured
customers, revenue, outcomes, or promises of wealth in this build. A credible next
milestone is a repaired task flow reviewed with people who use assistive
technology, with permission to publish its measured limitations and results.
The [consent-based pilot plan](PILOT.md) defines the proposed five-flow validation,
benchmark-before-training decisions, and business milestones. These milestones
are not achieved merely because the package builds or its synthetic tests pass.

## Reference standards and evaluation limits

Implementation and fixture code here are SZL-authored. Standards and outside
projects retain their own attribution and licenses:

- [W3C WCAG 2.2 Understanding Conformance](https://www.w3.org/WAI/WCAG22/Understanding/conformance.html): conformance needs evaluation beyond one automated rule.
- [W3C ARIA Authoring Practices patterns](https://www.w3.org/WAI/ARIA/apg/patterns/): reference interaction patterns, not proof this tool implements all patterns.
- [Deque axe-core](https://github.com/dequelabs/axe-core): an independent automated-testing project; this pilot does not claim to run axe-core unless a separate witnessed run is recorded.

This first release does not certify accessible names across arbitrary DOM trees,
keyboard behavior, focus management, contrast, screen-reader output, cognitive
usability, or whole-site/legal compliance. Those require additional tooling and
human evaluation, including people with disabilities.
