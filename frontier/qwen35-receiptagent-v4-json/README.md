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

### Exact train-token and loss-mask conformance

`training_tokens.check_training_tokens(snapshot, binding, encode, decode)` adds a
pure, train-only check between byte retention and a future admitted runner. It
checks all 360 examples sequentially against the declared immutable base,
tokenizer/template/policy digests and resource limits. The trusted adapter must
return the actual rendered text, character offsets, input IDs, attention mask
and unshifted collator labels. Complete assistant JSON and the exact declared
termination sequence must be supervised; prompt and right-padding positions
must be masked. Packing, truncation, ambiguous offsets and extra assistant text
fail closed. Python character offsets are required, not UTF-8 byte offsets.

The result is an immutable ordered hash ledger, not target text or token IDs.
Digests bind declarations and observed tensors but do not authenticate downloaded
assets, make hashes confidential or prove that another process consumed those
tensors. The future worker must freshly capture input, authenticate its runtime
and tokenizer, and retain/use the exact checked tensors or requalify them.
Callbacks are trusted Python code, not a sandbox. Fixed three-message framing is
deliberately restrictive; actual tokenizer compatibility is UNKNOWN until tested.
All training, publication and execution flags remain false. No model, tokenizer,
optimizer, network client, AirLLM runtime or GPU is imported or launched here.

`training_tokens.retain_training_tokens(snapshot, binding, encode, decode)` uses
the same checks and returns all 360 rows' exact immutable `input_ids`,
`attention_mask` and unshifted `labels` tuples together with the same redacted
`TokenReport`. Each row is encoded once; the returned tuples are the ones that
passed conformance and capacity checks. Any failure returns no partial result.
The adapter wrapper is detached before decoder callbacks, and mutable tensor
containers or non-integer values are rejected. Object representations omit the
tensors, but explicitly serializing the retained object exposes training content:
keep it private and serialize only `result.report` when a redacted report is
needed. A future separately admitted worker must consume these retained tensors
without retokenizing or modifying them. This trusted-process interface supplies
no model loader, optimizer, runtime qualification, admission or launch authority;
all eligibility and authority flags in its report remain false.

The software tests use SIMULATED character-tokenizer fixtures, including Unicode,
shared EOS/padding IDs and multi-token endings. Passing them is not a real-model
evaluation, runtime qualification or a training permit:

```sh
python -I -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json -p test_training_tokens.py -v
```

An optional offline base-tokenizer precheck exercises the committed **train
split only** with the seven pinned tokenizer/template files for the planned
`unsloth/Qwen3.5-0.8B` revision. It requires isolated `transformers==5.5.0`,
`huggingface-hub==1.5.0`, `tokenizers==0.22.2`, and `jinja2==3.1.6`. Supply a
local directory containing exactly those seven files; the script verifies
their byte hashes and loads a private copy in offline mode with remote code
loading disabled. Installed libraries are trusted Python code: version checks
are not library-byte authentication or an OS sandbox against reads/network.
The supplied tokenizer directory must be on a local fixed drive without
symlink/reparse-point components. It must be controlled by a trusted writer:
this precheck does not defend against concurrent hostile filesystem mutation.
It checks all 360 train rows for fixed framing, complete character offsets,
non-straddling assistant supervision boundaries, exact decoding, and the
declared ending IDs. It emits a redacted, locally `MEASURED` precheck record
with a chained token/offset digest and unsigned full-record checksum:

```sh
python -I -B frontier/qwen35-receiptagent-v4-json/probe_base_tokenizer.py \
  --tokenizer-dir /absolute/path/to/pinned/tokenizer-files
```

This is not `check_training_tokens` with a fresh Nemo-admitted snapshot and
actual trainer collator. It does not authenticate the model configuration or
weights, qualify GPU resources, run an optimizer or open dev/final-gate cases.
The record explicitly keeps training, publication and execution false. The
source binding is `DECLARED`; signatures and independent replay are
`UNAVAILABLE`. A different tokenizer asset byte set requires a reviewed binding
change; a different library build needs fresh qualification, even if its version
string matches.

### Pinned base-config boundary (separate offline precheck)

The optional `probe_base_config.py` reads only the seven pinned tokenizer files
and the exact `config.json` bytes from the **same** planned base revision. Pass
that config file separately; the tokenizer directory must still contain exactly
seven files. It checks the declared text architecture, context limit and two
vocabulary sizes against the loaded tokenizer, without importing weights or
opening any curriculum split:

```sh
python -I -B frontier/qwen35-receiptagent-v4-json/probe_base_config.py \
  --tokenizer-dir /absolute/path/to/seven/pinned/tokenizer-files \
  --model-config /absolute/path/to/pinned/config.json
```

At revision `23c69c53358a07516b5827588b3fdb12ae78fd65`, the text-model
config declares EOS ID `248044` (`<|endoftext|>`), while the chat tokenizer
declares EOS ID `248046` (`<|im_end|>`), and the train template terminates with
`248046, 198`. The model's padded vocabulary is 248320 entries, versus 248077
loaded tokenizer entries. These are observed, distinct declarations, **not** a
runtime generation stop-policy qualification or evidence of an incompatible
weight tensor. A future runner must bind its explicit stop/padding policy to its
actual loader and held-out decode; silently inheriting one EOS field is unsafe.
The output retains `runtime_stop_policy=UNQUALIFIED_REQUIRES_EXPLICIT_BINDING`
and all eligibility and authority flags false. Its `MEASURED` label covers only
this offline config/tokenizer readback; the asset-to-revision source binding is
`DECLARED`, unsigned and not independently replayed.

### Fresh train-only tensor reference precheck (not a trainer)

`probe_train_token_tensors.py` connects a **fresh** source-bound Nemo R1–R5
capture of the explicit 360-row train file to `check_training_tokens` using the
eight pinned local base files (seven tokenizer files plus separate `config.json`)
and the exact dependency versions above. It never opens dev or held-out cases.
The reference adapter renders with the actual `Qwen2Tokenizer`, builds unshifted
assistant-only labels, and checks every token/character offset, loss mask,
termination token and exact decode. The pinned chat template inserts an empty
`<think>\n\n</think>\n\n` block before assistant JSON even with
`enable_thinking=False`; the bound static framing includes it. Dropping that
block fails the full-render check.

Use a trusted local fixed-drive asset directory containing exactly the seven
tokenizer files, a separate exact `config.json`, and an explicit pinned Nemo
source checkout as described above. The local precheck caps are 2,048 tokens
per row and 400,000 tokens in total; they are **not** signed resource admission.

```sh
python -I -B frontier/qwen35-receiptagent-v4-json/probe_train_token_tensors.py \
  --tokenizer-dir /absolute/path/to/seven/pinned/tokenizer-files \
  --model-config /absolute/path/to/pinned/config.json \
  --train frontier/qwen35-receiptagent-v4-json/train.jsonl \
  --manifest frontier/qwen35-receiptagent-v4-json/curriculum-manifest.json \
  --nemo-source-root /absolute/path/to/pinned/szl-nemo-checkout
python -I -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json \
  -p test_probe_train_token_tensors.py -v
```

The emitted local `MEASURED` record is redacted to counts and hashes. It binds
the freshly checked train bytes and reference tensors, but not the future
trainer's collator, weight loader, GPU, generation stop policy, signer trust or
optimizer. The source-to-Hub asset binding remains `DECLARED`; signatures and
independent replay remain `UNAVAILABLE`. All eligibility/authority fields stay
false. A future trainer must use these exact retained tensors in its own
separately admitted runtime, or requalify its actual tensors; copying this
precheck record is never a launch permit.

## Signed sidecar integrity observations

`signed_admission.observe_admission(...)` checks one closed canonical ASCII
sidecar and its fixed Ed25519 domain plus NUL plus canonical payload bytes.
`requirements-admission.txt` deliberately aligns its `cryptography==50.0.0`
profile with existing Forge CI; it is not a latest-version claim, wheel-byte
authentication, runtime lock or security certification. The module has no
private-key, signing, file/network, nonce-consumption or launch interface.
An unavailable, unsupported or differently versioned backend leaves signature
verification NOT_RUN. Invalid signatures are reported as mismatches.

The caller must supply the complete expected canonical payload bytes, bounded
raw-public-key mapping, trust classification, clock, revocation state and maximum
age independently of the sidecar. `INDEPENDENTLY_PINNED` is a caller assertion
observed by this function; the module does not establish its provenance. Signature,
context, trust, time and revocation observations stay separate. The immutable
result cannot be used as a boolean and contains no permit or eligibility fields.
Even every matching observation cannot change the current false candidate policy
or establish owner approval, durable single use, containment or training authority.

The fixed positive and wrong-domain fixtures are SIMULATED software data signed
only with the already-public [RFC 8032 section 7.1 TEST 1 seed](https://www.rfc-editor.org/rfc/rfc8032#section-7.1).
The test-only `fixtures/generate_simulated_ed25519_fixture.py` prints deterministic
fixture data and never generates a key or writes a file; production imports none
of it. Its public test key must never enter an owner trust policy. The tests also
replay the published RFC primitive vector and retain the actual false candidate
flags. This raw-Ed25519 sidecar profile is distinct from canonical `szl-receipt`
DSSE/ECDSA-P256-SHA256 and v3 authenticated-receipt signatures; no format equivalence
or algorithm migration is claimed. [Maintained verification API](https://cryptography.io/en/50.0.0/hazmat/primitives/asymmetric/ed25519/).

```sh
python -I -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json -p test_signed_admission.py -v
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

The native worker and supervisor below are guarded source components, not an
admitted training launcher. Reusing the v3 launcher would bind the wrong
curriculum and targets. `training_eligible` deliberately stays false. The next
release phase requires separately reviewed admission/runtime compatibility, not
a manual override of the contract-only state.

## Native worker and supervisor: source implementation, launch BLOCKED

`native_worker.py` implements one train-only supervised-smoke update, not a full
training schedule. It freshly captures all 360 train rows through both curriculum
gates, retains the tokenizer's exact immutable IDs/masks/labels, verifies their
signed ledger and policies, and only right-pads with ignored labels. It sums
causally shifted supervised-token cross entropies across every microbatch and
divides by the global supervised-token count. Unequal-length row means are not
averaged equally. There is one gradient clip and one AdamW step after all 360
rows, with finite loss/gradient/parameter checks and exact optimizer coverage.
No dev or held-out input, generation, tool execution, Hub upload or signing API
is provided.

The explicit loader is `Qwen3_5ForConditionalGeneration`, with local safetensors,
no remote code and no fallback loader. The recipe is NF4/double-quant/bfloat16,
LoRA rank 16/alpha 32/dropout 0/no bias, and AdamW at 1e-4. Only the seven named
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` suffixes
under `model.language_model.layers` are eligible. This deliberately does not
claim adaptation of Qwen3.5's linear-attention projection modules. The exact
qualified full-name inventory, actual quantized module classes, frozen base,
trainable LoRA parameters and optimizer object identities must agree. Missing,
unexpected or mismatched loader keys and loader errors deny. Export is two
data-only LoRA files and a narrow REPORTED result, never a signed SKU or a model
evaluation. Local CPU math fixtures are not qualification of this GPU recipe.

`native_supervisor.py` implements fixed-command Linux systemd/cgroup-v2
supervision helpers: durable single-use nonce consumption, a release handshake,
actual unit/cgroup/resource-property readback, bounded process logs, telemetry,
memory, time, disk/output/artifact caps, cost qualification and owned-unit drain.
An occupied GPU denies; there is no stop, unload, eviction or sharing exception
for Bonsai or any other workload. The child gets no inherited Hub, SSH, Python
path or tracking credentials. The boundary is explicitly
`TRUSTED_SOURCE_BOUND_PROCESS_NOT_HOSTILE_SAME_UID_SANDBOX`; source hashes,
systemd property strings and a Python object are not a hostile-code sandbox.
Missing independently pinned barrier qualification or live readback denies.
The fsynced receipt chain provides byte integrity, not a signature or independent
witness. No NVML energy integration is implemented: joules remain UNAVAILABLE.

`native_runtime.py` accepts a closed request (`schema`, `paths`) and requires an
owner-policy SHA-256 supplied out of band, not by that request or its sidecar.
It rejects public RFC test pins, stale/revoked/mismatched admission observations,
unknown closure IDs, aliases/hardlinks, changing control files, unbounded limits
and qualification mismatches. A policy file calling itself independently pinned
does not establish that provenance. The current source's false eligibility flags
deny before reading model assets, consuming a nonce, probing telemetry, spawning
a process or importing model frameworks. Frozen manifest/conformance/token
reports remain non-authorizing and are not rewritten as permits.

There is also an explicit **CONTAINER_IDENTITY_UNAVAILABLE** hold even if a
future protected source transition makes the candidate eligible. The existing
v1 signed profile binds `runtime.container_sha256`; this native systemd runner
has no live qualified OCI-identity adapter. A native filesystem-tree hash must
not be relabeled as that container identity. A governed versioned native-runtime
profile or a real measured container adapter, exact dependency/runtime identity,
independently pinned owner admission, dedicated available compute and supervised
qualification remain required. The production loader intentionally returns no
`BoundRun` while this compatibility hold exists. Helpers exercised with synthetic
objects are not a route around it. There is no CLI override.
The worker also has a separate **DEPENDENCY_IDENTITY_UNAVAILABLE** hold: no
installed wheel/package-tree identity adapter has yet been qualified. Version
strings, a wheel lock file or a qualification record describing its own digest
do not authenticate the actually imported dependency bytes. Neither hold is
cleared by unit tests, property strings or an owner-policy label.

Current command interfaces (not launch authorization):

```sh
python -I -S -B frontier/qwen35-receiptagent-v4-json/native_supervisor.py \
  --check-readiness --bound-run /absolute/request.json \
  --owner-policy-sha256 OUT_OF_BAND_VERIFIED_SHA256
python -I -S -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json -p 'test_native*.py' -v
```

`--supervise-smoke` uses the same mandatory denial path. The worker additionally
is dispatched by the supervisor with the exact `-I -S -B` startup flags, requires
the exact supervisor stdin marker and independently rechecks its
cgroup and bound context. These interfaces do not rent compute or launch an HF
Job. Full training, frozen 180-case evaluation, independent replay, publication
to a new candidate Hub identity and end-to-end Agent Studio acceptance are
separate, still-unavailable release stages. AirLLM remains a separately studied
low-memory inference option, not a training method, quality result or deployment
proof. No signed predecessor model is changed by these components.

### Native file-byte identity prerequisite

`native_identity.verify_native_file_manifest(root, manifest_path,
expected_manifest_sha256=...)` compares an exact directory inventory with a
separately supplied manifest digest. The closed canonical JSON profile is
`szl.receiptagent.native-file-manifest/v1`, with sorted `files` entries containing
only `path`, `size_bytes` and `sha256`. The manifest must live outside the tree.
Regular files are streamed under fixed file/total/count limits; missing or extra
files, extra empty directories, aliases, hardlinks and observed changes deny.
Startup files and native libraries are treated as opaque bytes, never imported
or executed. Synthetic tests also exercise an isolated `-I -S -B` subprocess.

This is a data comparison prerequisite, not an installed-wheel provenance or
complete-runtime authentication adapter. The caller still needs a reviewed
complete dependency inventory, authenticated interpreter/stdlib/native libraries,
a qualified bootstrap before Python site initialization, read-only consumption,
and the separate native admission profile. The verifier assumes a trusted single
writer, not hostile concurrent filesystem control. It does not supply a production
manifest, owner pin, signature, permit, launch route or import-path change. Both
existing runtime identity holds and all candidate flags stay unchanged.

```sh
python -I -S -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json -p test_native_identity.py -v
```

### Pinned wheel byte comparison prerequisite

`native_wheel_identity.verify_native_wheel_inventory(wheel_root, inventory_path,
installed_root, native_manifest_path, expected_inventory_sha256=...,
expected_native_manifest_sha256=...)` compares a separately pinned wheel set with
the pinned native manifest and actual directory. The closed canonical profile is
`szl.receiptagent.native-wheel-inventory/v1`, containing `schema`,
`native_manifest_sha256`, and sorted `wheels` entries with only `path` (a wheel
basename), `size_bytes`, and `sha256`. Both external pins are mandatory; a digest
inside the inventory is not a trust anchor. Results are immutable counts and
digests, not a capability to import, launch, train or serve.

This deliberately narrow profile supports exact root-layout copies of ZIP32
STORED/DEFLATED wheels. It rejects `.data` relocation, rewritten metadata,
generated wrappers/bytecode, unexplained installed files, encrypted or special
entries, ambiguous paths, overlapping file destinations and unsupported archive
features. Directory namespaces may be shared only with disjoint file paths.
Every file, including `.dist-info/RECORD`, startup files and native libraries, is
compared as opaque bytes. `RECORD` and distribution metadata confer no trust.
There is no installer, extraction, package discovery, dependency resolution,
network access, or package import. Standard installer transformations are not
silently exempted; this is not a generic audit of an arbitrary pip environment.

Archive hashes are checked before ZIP parsing. Raw central-directory and
local-header interpretation and metadata limits are checked before constructing
`ZipFile`; payload reads are separately bounded. No ZIP64, multi-disk archives,
comments, extras, data
descriptors, preambles, gaps or trailing overlays are accepted by this version.
The same verified file handle is used through parsing and streaming, with
change checks under the trusted-single-writer assumption. This still does not
establish immutable consumption or hostile-same-UID containment.

No production wheel inventory is supplied or qualified here. Complete dependency
selection, compatible wheel tags, publisher provenance, interpreter/stdlib/native
library identity, a pre-site bootstrap and separately trusted native admission
remain required. Both production identity holds and candidate flags stay unchanged.

```sh
python -I -S -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json -p test_native_wheel_identity.py -v
```

### Pre-site, verify-only bootstrap

`native_bootstrap.py` is a standalone byte-observation entry point. Start it
with `-I -S -B`: isolated startup, no automatic `site` initialization, and no
bytecode writes. It first checks the actual interpreter flags, then captures
and checks both adjacent helper source files against separately supplied
SHA-256 pins before executing either retained source buffer. It does not load
helper bytecode, search for alternative helpers, add package paths, run `.pth`
files, or import the candidate dependency tree. All supplied input paths must
be absolute and pass the helpers' bounded plain-file/tree checks.

```sh
python -I -S -B /absolute/source/native_bootstrap.py \
  --wheel-root /absolute/wheels \
  --inventory /absolute/wheel-inventory.json \
  --installed-root /absolute/installed-tree \
  --native-manifest /absolute/native-manifest.json \
  --inventory-sha256 OUT_OF_BAND_WHEEL_INVENTORY_SHA256 \
  --native-manifest-sha256 OUT_OF_BAND_NATIVE_MANIFEST_SHA256 \
  --native-identity-sha256 REVIEWED_NATIVE_IDENTITY_SOURCE_SHA256 \
  --native-wheel-identity-sha256 REVIEWED_NATIVE_WHEEL_IDENTITY_SOURCE_SHA256
python -I -S -B -m unittest discover -s frontier/qwen35-receiptagent-v4-json -p test_native_bootstrap.py -v
```

A successful observation means only that the supplied wheel/tree bytes matched
the supplied commitments during this invocation. The pins must come from a
separately trusted source; generating pins from arbitrary local files does not
authenticate those files. The bootstrap itself, executing interpreter, standard
library, native loader and prior startup environment remain externally trusted
inputs, not identities established by the bootstrap. In particular, rejecting
an invocation without `-S` cannot undo startup code that already ran. Mutable
files are not made immutable by successful hashing. The trusted-single-writer
assumption remains in force.

This entry point never returns a `BoundRun`, consumes a nonce, launches a worker,
loads a model, touches the GPU, or grants training/publication/execution authority.
The existing worker and supervisor do not yet consume it as an admission gate;
they only enforce the observed interpreter startup flags before path-dependent
runtime imports. Both production identity holds and all candidate flags remain
unchanged.
Software tests use SIMULATED wheel/package fixtures, not a qualified production
runtime or model evaluation.

### Remaining native admission bindings

Native admission requires a separately versioned native profile and signature
domain. The current container profile must remain unchanged: a native tree hash
cannot fill `runtime.container_sha256`. Before any production `BoundRun` can be
returned, the signed context must be compared with actual observed inputs:

- Candidate and publishing-policy bytes, exact executable source closure,
  supervisor policy and canonical worker-closure commitment.
- Every contract, preregistration, gate, harness, schema, validator and Nemo
  source commitment, plus the exact loader/architecture semantics.
- Complete transitive installed dependencies and wheel/platform qualification,
  authenticated interpreter/stdlib/native libraries, controlled imports, and
  qualified read-only consumption after pre-site verification.
- Actual runner/device identity, fresh containment and resource qualification,
  evidence/receipt/durability/drain/terminal/anti-replay policy commitments.
- An independently pinned training public key with trust provenance, current
  revocation/freshness policy and an owner-signed bounded smoke admission.

These are unfinished bindings, not reachable training bypasses: current source
eligibility and the unconditional identity holds still deny. Neither a Git/SSH
key nor the public RFC test key may be repurposed as the training signer. The
occupied-GPU denial also stays in force; keeping Bonsai running requires a
separate, exclusively admitted compute allocation for ReceiptAgent training.

## Prior art

This uses existing open standards and maintained software, not a claimed new
JSON format or proof system:

- [JSON Schema validation, Draft 2020-12](https://json-schema.org/draft/2020-12/json-schema-validation).
- [Python JSON decoder hooks and finite-number handling](https://docs.python.org/3/library/json.html).
- [jsonschema validation and format checking](https://python-jsonschema.readthedocs.io/en/stable/validate/).
- [PyPA wheel layout and installation transformations](https://packaging.python.org/en/latest/specifications/binary-distribution-format/).
- [Python ZIP archive behavior and decompression pitfalls](https://docs.python.org/3/library/zipfile.html).
- [Existing AirLLM qualification package-byte comparison](../airllm-qualification-lab/local_airllm.py), a studied comparison pattern rather than a training or runtime admission result.

The SZL-specific contribution is the versioned proposal/withhold contract,
fail-closed request/evidence bindings, and preservation of historical evidence.
