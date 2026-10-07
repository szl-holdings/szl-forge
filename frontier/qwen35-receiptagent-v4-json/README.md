# ReceiptAgent v4 JSON-only contract

Evidence class: DECLARED source contract. This directory contains software,
not trained weights. Model training, model evaluation, independent replay and
publication remain BLOCKED or UNAVAILABLE as recorded in `candidate.json`.
Passing software conformance tests does not change those states.

This is a new protocol identity and planned candidate, not a relabeling of a
published adapter or a repair of the frozen v3 evaluation set. The three v3
split files and their existing schemas remain unchanged. No Hub repository
is created or written by this contract.

## Observable protocol

Every model target is intended to be one JSON object. Plain `REFUSE:` lines,
Markdown, trailing prose, duplicate members, non-finite numbers, unpaired
Unicode surrogates, extra fields and oversized structures are rejected.

- DRAFT: `PROPOSE_ONLY`, supplied evidence all `OK`, evidence copied exactly,
  fixed non-narrative claim, all required observable checks recorded.
- RECOVERY: `PROPOSE_ONLY`, one failed supplied evidence status, exact bound
  recovery code/evidence ID/failed check, no execution.
- REFUSAL: every other recognized authority, empty evidence, null recovery,
  case-bound blocked action, failed AUTHORITY check, no secret-literal echo.

All branches preserve `approvalRequired=true`, `executed=false`,
`autonomyEligible=false`, receipt status `NOT_BOUND`, and external-controller
validation/approval/execution. Conformance never grants action authority,
authenticates evidence, or creates a signed receipt. Fixed claim strings avoid
accepting arbitrary narrative assertions under a structurally valid JSON shell.

The request includes `contractVersion=szl.receiptagent.json-only/v1` and a
`train-json-v4-...`, `dev-json-v4-...` or `test-json-v4-...` identity. Future
identity construction includes protocol version, split, response kind, task
family and canonical content; changing kind no longer reuses the same identity.
An ID namespace alone does not establish split independence.

## Schema binding and validation

`json_contract.py --schema request` and `--schema response` export Draft 2020-12
schemas. The compiler extends the two v3 schemas without editing them. Exact
SHA-256 bindings in the module and `candidate.json` fail closed on parent drift;
changing a parent requires a separately reviewed version/binding change.
All `$ref` values are local fragments; no remote schema is loaded.

Validation independently recomputes authority/evidence routing, identity,
validation effort, evidence identity/copying, recovery and self-check coverage.
Every response must copy the controller-supplied `requestSha256` from
`prompt_envelope(request_text)`. The validator recomputes this SHA-256 over the
complete canonical request, including task and optional forbidden literal;
changing those fields cannot replay a response by keeping the same request ID.
This digest is an equality binding, not a signature, origin authentication or
confidentiality mechanism. Do not publish sensitive request hashes or put real
secrets in test fixtures. Identifier, digest and endpoint patterns require the
absolute end of the string, including rejection of a trailing newline.
It does not trust the model's declared PASS/FAIL as an authorization verdict.
The runtime parses finite UTF-8 JSON with byte/depth/node limits and validates
timestamps without depending on optional format extras.
The timestamp profile requires uppercase `T`/`Z`, valid calendar dates and
bounded hour/minute/second fields; it does not support leap-second timestamps.

Limits are 65,536 UTF-8 bytes per document, depth 32 and 2,048 value nodes. The
literal echo check uses decoded Unicode strings and case-folding; it is not a
semantic information-flow guarantee. `OK`/`STALE`/other evidence statuses are
supplied inputs: their truth and freshness must be checked by the controller.
Routing uses the typed `requestedAuthority`, not semantic interpretation of
`task`. A mislabeled `PROPOSE_ONLY` task can conform structurally but still has
no execution authority. Actual intent classification and policy review remain
external-controller obligations; this is not a semantic refusal benchmark.
No cryptographic receipt validation, retrieval, tools, model import, GPU work,
training, serving, signing or file mutation is performed by the validator.

## Run offline

Use an environment with `jsonschema==4.26.0`. No model download or API key is
needed for protocol checks. From the repository root:

```sh
python -I -B frontier/qwen35-receiptagent-v4-json/json_contract.py --schema response
python -I -B frontier/qwen35-receiptagent-v4-json/json_contract.py \
  --request /absolute/path/request.json --response /absolute/path/response.json
python -I -B -m unittest discover \
  -s frontier/qwen35-receiptagent-v4-json -p 'test_*.py' -v
```

CLI exit 0 means schema/pair conformance only. Exit 1 means BLOCKED input or
binding; exit 2 is invalid CLI usage. Neither success nor failure permits
training, publication or execution. Software tests use SIMULATED test
targets, not actual model decodes or held-out model results.
The existing blocking offline CI lane owns these tests; no publisher was added.

## Fresh curriculum and preregistration

The v4-only curriculum has 360 train rows, 90 dev rows and 180 separately authored
public frozen final-gate rows. Families are disjoint (four/two/three), and each
split balances DRAFT, RECOVERY and REFUSAL. Each train family has six examples of
every recovery status and blocked authority; dev has three and final has four.
All evidence and reference targets are synthetic. `curriculum-manifest.json`
commits exact LF bytes; `preregistration.json` declares the future denominators,
limits and remaining launch/release gates. This is not a signed curriculum.

`generate_curriculum.py` authors only train/dev. It reads final-gate aggregate
commitments, never final-gate cases/templates. A separate custodian authored the
final gate. A non-authoring reviewer checked every split for exact and normalized
task/value overlap; the independent integrity test freezes those checks and
commitments. Zero textual overlap is not a semantic-independence proof: the fixed
protocol, claims and output rules are deliberately shared. The final 180 cases
have 75 normalized task forms and 12 normalized evidence-value forms, not 180
independent semantic problems. Access is `PUBLIC_FROZEN_NOT_BLIND`.

This is a typed routing/binding/copying benchmark, not software analysis, intent
classification, or a comparable replacement for old v3, Khipu or Chaski scores.
Train/dev evidence counts are zero through two; final counts extend through four.
Exact tokenizer capacity and nontruncating runtime qualification remain UNKNOWN.

```sh
python -I -B frontier/qwen35-receiptagent-v4-json/generate_curriculum.py --check
```

Generation refuses differing existing artifacts; `--write` creates only missing
matching artifacts, and final readback rejects racing substitutions. These are
owner-controlled single-writer reproducibility utilities, not authorization
boundaries. Missing/drifted committed artifacts fail `--check`. Never regenerate
or retune a final gate after seeing a model failure; freeze a new experiment.

## Train-only conformance (not training authorization)

`curriculum_admission.py` reads the explicit training file and exact immutable
manifest only; it does not discover/glob/open dev or final-gate cases. It checks
strict JSON, all pair/ID/digest bindings, SIMULATED labels, exact family/class and
status/authority coverage, and pinned LF contract/gate source bytes. It snapshots
the same bounded, non-aliased training bytes into a private temporary directory
and reruns the real generic validator and Nemo R1-R5 gate. The maintained Nemo
verifier checks the complete unsigned receipt chain and each exact prompt/target
input hash; a saved report or supplied signature is never accepted as a shortcut.

Use an environment with `jsonschema==4.26.0` and an explicit source checkout of
the [reviewed doctrine kernel revision
f7b33e1b1fe8f3b9b729a27567bcba375e2e0d6e](https://github.com/szl-holdings/szl-nemo/commit/f7b33e1b1fe8f3b9b729a27567bcba375e2e0d6e).
The `nemo_source_binding.py` software profile
`szl.receiptagent.v4-nemo-source-binding/v1` verifies SHA-256 commitments for six
kernel modules before private imports. It does not resolve an ambient installed
`szl_nemo` package. Canonical JSON uses the reviewed kernel's own local fallback,
not an optional ambient `evidence_core` installation. The source root must be the
repository directory containing `szl_nemo`, not the package directory itself.
An absent source root is NOT_READY, never permission to import the ambient kernel.

Acquire the source in a new directory; do not reuse or reset another writer's
checkout. The following commands do not install or execute that repository:

```sh
git clone --no-checkout --config core.autocrlf=false https://github.com/szl-holdings/szl-nemo.git .nemo-v4-source
git -C .nemo-v4-source sparse-checkout set szl_nemo
cd .nemo-v4-source
git checkout --detach f7b33e1b1fe8f3b9b729a27567bcba375e2e0d6e
cd ..
```

Preserve the committed LF bytes: automatic checkout line-ending conversion can
change the module hashes and will fail source admission. For archive exports,
use command-local `git -c core.autocrlf=false -c core.eol=lf archive` as well;
default `git archive` can apply the local line-ending configuration.

The source-only CI lane obtains the same revision in `.nemo-v4-source` using a
separate pinned checkout action with `persist-credentials: false`. The legacy
shared gates retain their existing kernel installation; that installation is
not used as the v4 source binding. Run the explicit v4 conformance check:

```sh
python -I -B frontier/qwen35-receiptagent-v4-json/curriculum_admission.py \
  --check-conformance \
  --train frontier/qwen35-receiptagent-v4-json/train.jsonl \
  --manifest frontier/qwen35-receiptagent-v4-json/curriculum-manifest.json \
  --nemo-source-root .nemo-v4-source
```

The earlier 360-row replay against the older kernel cannot be reused as evidence
for this revision. Rerun train-only conformance with this source binding and
retain its exact source commitments and results. The source-byte check and
private import profile are software integrity controls, not a sandbox against a
hostile Python process, authenticated source origin, independent witnessing or
training authorization. No model scores are established by a source replay.

The CI `probe_nemo_binding.py --nemo-source-root .nemo-v4-source` exercises four
SIMULATED fixtures against the real captured kernel: scientific numeric tokens,
a snake-case JSON metric, adjacent non-English prose around an English metric,
and an explicit UNKNOWN answer. It also checks receipt-chain tampering. These
are source regressions, not a model evaluation. Exploratory `accuracy_0.91` and
`0.91精度` answers remain unrecognized by the pinned lexical checker; the probe
reports these observations separately with no passing-test credit. This kernel
does not establish general multilingual or semantic claim detection.

CI retains the probe and train-only conformance JSON reports as a separate
artifact because the complete receipt-chain report can exceed a log viewer's
single-line limit. The artifact is named `v4-source-conformance-<run_id>` and
contains only `numeric-probe.json` and `curriculum-conformance.json` from the
runner's temporary `v4-source-conformance` directory. Only these two explicitly
named report files are uploaded; no curriculum, model, checkout directory or
wildcard is included. The pinned upload step runs even after failure so that
available diagnostics can be retained for 30 days. A failed probe or conformance
command still fails its gate step; report retention never converts that failure
to a pass. If an earlier command fails, later reports may be absent: inspect the
exact run/commit, gate conclusion and report contents rather than treating an
artifact's existence as complete conformance evidence. Retained unsigned reports
are not independent witnessing, signed curriculum approval or training authority.

Exit 0 means train-only conformance, not authority. Exit 1 is invalid/drifted
source/data or failed integrity. Exit 2 means a required gate/kernel/verifier is
NOT_READY (or invalid CLI usage), never a pass. Success still reports
`state=BLOCKED`, all eligibility/authority false, signature invalid/unavailable
and runtime binding unavailable. Its unsigned chain is software integrity evidence,
not an owner-signed curriculum, authenticated origin or permission to train.
Actual training/evaluation remains NOT RUN, optimizer steps zero, energy UNAVAILABLE.

### Retained train input for a future source-bound runner

`capture_train_only(train_path, manifest_path, nemo_source_root=...)` runs the
same fresh checks and returns `TrainOnlySnapshot` only after they succeed. Its
immutable manifest and train buffers retain the exact checked input bytes; its
report buffer is the canonical serialization generated by that conformance run.
A failed or unavailable gate raises instead of yielding a snapshot;
the existing CLI keeps its partial diagnostic report and exit-code behavior.

`training_data.training_messages(snapshot)` converts that retained train buffer
into immutable system/user/assistant role-content triples. It binds the frozen
manifest and train hashes and checks the conversion shape without rereading
source paths or rerunning gates. It treats the full report as opaque provenance,
not a contract document or permit. The consumer imports no model runtime and
provides no tokenizer, optimizer, GPU, Hub, signing or launch interface.

This is a trusted-process byte-retention interface, not an unforgeable capability
or admission format. A future admitted runner must call fresh capture itself,
consume these same buffers and separately satisfy every signed-admission and
resource/supervision gate below. Deserializing a snapshot or substituting a saved
report is not an allowed shortcut. All eligibility and authority remain false.

The added capture/conversion tests use independently constructed SIMULATED rows;
they do not open actual dev or final-gate cases or establish model performance:

```sh
python -I -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json -p test_train_snapshot.py -v
python -I -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json -p test_training_data.py -v
```

## Still required before training

1. Obtain protected source merge/readback and owner-signed curriculum admission;
   immutable source references and peer textual checks are not authorization.
2. Keep final-gate cases out of gradients and development. Do not inherit the old
   72-case/+15 result threshold as if this were the same task or assert semantic
   independence from identifiers or textual checks alone.
3. Bind a signed curriculum to the generic SFT validator, reviewed immutable
   Nemo R1-R5 gate, this JSON contract and a source-current trainer/supervisor.
4. Establish bounded local resources, emit chained per-run evidence and verify
   a supervised smoke before any full training attempt.
5. Run the frozen held-out gate and independent replay. A new candidate uses a
   new Hub identity; signed predecessor SKUs stay intact. No promotion is
   inferred from loss, schema checks, file presence or a serving HTTP response.

No training launcher is supplied here: reusing the v3 launcher would still bind
the v3 curriculum and targets. `training_eligible` deliberately stays false.
The next source phase is a separately reviewed, fully bound curriculum/runtime,
not a manual override of the contract-only state.

## Prior art

This uses existing open standards and maintained software, not a claimed new
JSON format or proof system:

- [JSON Schema validation, Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-validation).
- [Python JSON decoder hooks and finite-number handling](https://docs.python.org/3/library/json.html).
- [jsonschema validation and format checking](https://python-jsonschema.readthedocs.io/en/stable/validate/).

The SZL-specific contribution is the versioned proposal/withhold contract,
fail-closed request/evidence bindings, and preservation of historical evidence.
