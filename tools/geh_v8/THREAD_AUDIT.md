# Thread audit — Chaski r4 (turns 1–45) and GEH v3–v7 (turns 46–50)

Audited 2026-09-30 against the committed state of `szl-holdings/szl-forge`, the `lutar-lean`
toolchain pin, the Qwen3.5-0.8B chat template on the Hub, and the upstream Lean REPL / PyPantograph
sources. Everything below that says "verified" was executed in this session; hashes are listed so
you can re-derive them.

## Executive summary

1. **The thread's central open item was never a harness bug.** The "19-token truncation" receipt
   (0/5 drafts, 3/6 refusals) measures the *original* r4 adapter (dir-hash `b116832a…`). A
   *retrained* r4 adapter (weights `f1a2cdc3…`, trained 2026-09-17T02:36Z on a fixed curriculum)
   scored **5/5 + 6/6** on the same held-out gate four minutes later — and that receipt has been
   committed in `szl-forge` since 2026-09-17 (`bcdd1d85`, "three perfect models; r4-local 5/5+6/6
   MEASURED"). Turns 18–45 compared replays of the retrained adapter against the receipt of the
   original one and concluded "model innocent, harness guilty". That thesis is not supported.
2. **One gate is still genuinely open:** the canonical receipt path still holds the failing
   receipt A, and the passing receipt B was produced in the unpinned global Python 3.12 environment
   (torch 2.10.0+cu130) — the environment the thread told you *not* to use. r4 stays
   `NOT_PROMOTABLE` until the retrained adapter passes the canonical runner in the pinned runtime
   with the r2 control reproduced in the same run. The runbook does exactly that, and nothing else.
3. **GEH v3→v7 were never executed, never compiled, and as rendered in the thread could not run.**
   Turns 48–50 contain zero `[0]` index expressions where the code needs at least a dozen (chat
   rendering ate them as citation markers), v7 has a `SyntaxError`, its validator checks the wrong
   tool name (so a BLOCKED `sorry` would never be detected), its PyPantograph API was wrong, and the
   emitted "guard" was a one-line stub. Full table in Part B.
4. **GEH v8 is built and verified here**, not pasted: `GehGuard.lean` compiles under
   `leanprover/lean4:v4.18.0` (your lutar-lean pin), three red tests fail the build on `sorryAx`,
   `Lean.ofReduceBool` and a foreign axiom, the Lean REPL kernel loop closes proofs, the audit
   channel extracts tactic invocations and runs `#geh_guard`, a K=10 randomized crypto suite scores
   10/10 kernel-verified, receipts are DSSE-signed and chained, a zero-dependency verifier
   re-derives the gate from evidence, and 11/11 tests pass.
5. **Process finding:** never copy code out of chat. Every payload in this thread went through a
   renderer that strips `[0]`. Ship files with hashes (this bundle does) or nothing.

## Part A — Chaski r4: what the evidence says

### A.1 Two receipts, two adapters

| | Receipt A (canonical path) | Receipt B (four-way) |
|---|---|---|
| File | `chaski_r4/bakeoff_canonical_four_way_r4.receipt.json` | `chaski_r4/bakeoff_four_way_receipt.json` |
| computed_at | 2026-09-16T12:47:25Z | 2026-09-17T02:41:05Z |
| torch | 2.11.0+cu128 | **2.10.0+cu130** (global Python 3.12) |
| r4 adapter dir-hash | `b116832a34628407…` (original, plan dataset `8965f391…`) | `e1abc37a5c41a82b…` (retrained) |
| r4 result | **0/5 drafts, 3/6 refusals** (19-token outputs) | **5/5 drafts, 6/6 refusals** |
| chaski-5050 | 0/5, 6/6 (hash `b077cafe…`) | 5/5, 6/6 (hash `fc7da61d…`) |
| r2 control | 5/5, 6/6 (`e35df3be…`) | 5/5, 6/6 (`e35df3be…`) |
| base | 0/5, 6/6 | 0/5, 6/6 |
| gate hashes | identical | identical |
| file sha256 | `424f4fdd…4ccd0` | `2896db3f…a7c0` |

Both receipts are in the same commit, `bcdd1d85` (2026-09-17T10:46Z). `training_receipt.json`
(computed 2026-09-17T02:36:42Z) records the retrained adapter: dataset `0fea0d85…`, 40 rows, loss
0.982, `adapter_weights_sha256 = f1a2cdc3…`. That is the exact weights hash turn 44's payload
expected for the *local* r4 adapter — i.e., the adapter you were replaying on Sep 18 was the
retrained one, while the receipt you were reading (A; local copy sha `424f4fdd…`, byte-identical
to the committed file) was the original one's.

The dir-hash formula is the runner's own: `sha256(filename ‖ NUL ‖ bytes)` over sorted
`*.safetensors`. The runbook computes it for the adapter on your disk and tells you which of the two
it is (`--dirhash`).

### A.2 Why the 19-token theory looked plausible

* The Sep 18 replay used a *different prompt* (a user turn that lists all seven keys, no system
  message) than the held-out gate (a system prompt that names only `decision`, `approvalRequired`,
  `executed`). The original adapter emitting exactly those three keys and closing the object is a
  complete generation, not a truncation.
* The chat template cannot be the culprit: Qwen3.5-0.8B's template emits the same generation
  prefix (`<|im_start|>assistant\n<think>\n\n</think>\n\n`) for `enable_thinking=False` **and** for
  the undefined default, and the r4 trainer rendered the final assistant turn the same way. Verified
  by reading `chat_template.jinja` on the Hub.
* The Sep 18 "budget patch + canonical rerun" block **failed at its own verification step**
  ("expected 128, got 0") and never re-ran the gate; the pristine diagnostic trace was never run.
  No canonical evaluation happened on Sep 18 at all.

### A.3 Residual, honest uncertainty

Receipt B is a real MEASURED pass — but in an unpinned interpreter with a different torch/CUDA
build, and it was written to a non-canonical filename. Two adapters (r4 and 5050) flipping between
runs while r2 is stable is *also* consistent with knife-edge greedy decisions that differ across
CUDA builds. The bounded action that resolves both readings at once: run the pristine canonical
runner (`f4ca282a…dadb`) on the retrained adapter in `.venv-ra-v2` with the r2 control. If it
passes, promote that receipt to the canonical path by an explicit, reviewed copy. If it fails with
the same short outputs, environment sensitivity becomes the live hypothesis and the next harness
upgrade is to log the top-2 logit margin at the `}`-vs-`,` decision token per case.

### A.4 Other findings in the committed state

* `training_plan.json` still carries the *original* dataset hash (`8965f391…`) while
  `training_receipt.json` carries the fixed one (`0fea0d85…`). Update the plan or add a v2 plan.
* `training_plan.json.validation.semantic_gate_paraphrase_review = REQUIRES_HUMAN_REVIEW`. Under
  the qualification doctrine this is an unresolved contamination gate for the retrained curriculum
  (`exact_heldout_text_overlap` passed; semantic review did not happen).
* The archived runner (`evidence/r4_runner_20260916_091416.py`) used
  `AutoProcessor`/`AutoModelForImageTextToText`; the committed one uses
  `AutoTokenizer`/`AutoModelForCausalLM`. Receipt A may have been produced by either. Receipts do
  not record `transformers`/`peft` versions or the model class — add both (the runbook prints
  them before the gate).

### A.5 State block

```text
Artifact: chaski-r4 local adapter (retrained; weights f1a2cdc3… if the runbook confirms)
Canonical source: szl-holdings/szl-forge chaski_r4/ @ bcdd1d85 (pristine runner f4ca282a…dadb)
Training: TRAINED_CHALLENGER (training_receipt.json 2026-09-17T02:36Z)
Evaluation: MEASURED but SPLIT — A (original adapter, torch 2.11.0+cu128): 0/5, 3/6 | B (retrained, torch 2.10.0+cu130): 5/5, 6/6
Publication: UNPUBLISHED (chaski-r4 correctly absent from the Hub)
Promotion: NOT_PROMOTABLE
Blocking gate: canonical receipt path holds receipt A; no pinned-env canonical run of the retrained adapter with r2 control in-run; semantic paraphrase review open
Next bounded action: geh_v8_runbook.ps1 -RunCanonicalGate (writes a NEW receipt; promotion by explicit copy after review)
Protected state: receipts A and B, both adapters, gate files, pristine runner — untouched
Completion evidence: chaski_r4/evidence/canonical_rerun_<stamp>.receipt.json with r2 5/5+6/6 and r4 5/5+6/6 in the same run
```

## Part B — GEH v3–v7 defect table (turns 46–50)

| # | Finding | Severity | Where | v8 status |
|---|---|---|---|---|
| 1 | Code rendered with **all `[0]` indexes stripped** (turns 48–50 contain none; `st["goals"]`, `lines`, `split(None,1)`, `env["signatures"]["sig"]` all need one) | blocker | v5–v7 | files shipped, not chat text; hashes in manifest |
| 2 | `emit_a12` missing a closing paren → **SyntaxError**, whole file unimportable | blocker | v7 | compiled + tested |
| 3 | `validate_proof_trace` matches `lean.run_tac` while the loop emits `lean.run_tactic` → BLOCKED/EVIDENCE never detected → **fail-open** | critical | v7 | one tool vocabulary; `sorry_blocked_and_void` self-check |
| 4 | DSSE `verify` indexes `env["signatures"]["sig"]` on a list → TypeError → roundtrip self-check fails → no receipt could ever be issued | critical | v6–v7 | verified roundtrip + tamper rejection; pure-Python verifier cross-checked against RFC 8032 |
| 5 | `GUARD` constant was a one-line comment; `--emit-lean` wrote a stub | critical | v5–v6 | real guard, compiled under v4.18.0, three red tests |
| 6 | PyPantograph API wrong: `goal_tactic(state, goal_id=…, tactic=…)`; real signature is `goal_tactic(state, tactic, site)` (server.py) and `pip install pantograph` is an unrelated HTML5-canvas package | critical | v6–v7 | API corrected; transport marked NOT exercised here |
| 7 | `ScriptedProver.i` never incremented → infinite `intro h` | major | v7 | fixed; provers return candidate lists |
| 8 | `rng.randrange(1000)[997]` → TypeError in suite generator | major | v7 | rewritten generator with Python oracles |
| 9 | v4 leak check compared against an empty list (always passed) | major | v4 | regression test `contamination_scoring_detects_leak` |
| 10 | Axiom guard never invoked by the loop; v7 `guard_ok = any(... or True)` | critical | v5–v7 | PASS requires `#geh_guard allowed=true` via the audit channel |
| 11 | "Proof tree" was a linear loop; `done` checked only root children | major | v7 | candidate backtracking on immutable kernel states; dead leaves recorded |
| 12 | Lean snippets never compiled (`m!"{{…}}"` escaping, `collectAxioms` lifting unverified) | major | v5–v7 | compiled; JSON built with `Lean.Json` |
| 13 | No resource discipline: an unbounded REPL froze this 2-core sandbox by accumulating environments | major | (found building v8) | one base env, prelude env, per-RPC timeout → dead branch + respawn, RLIMIT_AS on POSIX |
| 14 | Version inflation v3→v7 across five turns with zero executions; each turn re-pasted ~600 lines | process | all | one tested file + tests + CI-style self-check in every receipt |
| 15 | LeanDojo transport asserted as working; needs a traced `LeanGitRepo` | honesty | v5–v7 | adapter kept, mock-exercised only, `verified_here=false` in receipts |
| 16 | Receipts asserted EU AI Act Art. 12 compliance | honesty | v3–v7 | labelled engineering mapping, not legal advice |

## Part C — What v8 actually does (verified in this session)

* **Kernel path:** `leanprover-community/repl` @ v4.18.0 as a Lake dependency of `geh-lean`, run via
  `lake env <repl>`; `sorry` opens a proof state, each tactic is a governed tool-call with pre/post
  hashes, goals come back pretty-printed, states are immutable (true backtracking).
* **Three independent checks for PASS:** live channel closes all goals; reconstructed file
  compiles with 0 errors / 0 sorries (`allTactics` extraction = kernel-verified tactic triples);
  `#geh_guard` reports axioms ⊆ {`propext`, `Classical.choice`, `Quot.sound`}. `native_decide`
  (`Lean.ofReduceBool`), `sorry` (`sorryAx`) and user axioms fail the build.
* **Contamination-resistant suite:** five crypto-property families (PAE framing, framing
  unambiguity, XOR-OTP involution, hash-chain step, nonce uniqueness), per-run randomized constants,
  canaryed binder/theorem names, `->`/`→` and spacing variance; Python oracle vs kernel `rfl/decide`.
  Surface variants are asserted to elaborate to byte-identical goals (kernel normalizes the surface).
* **Receipts:** DSSE (ed25519) over canonical JSON, keyid = sha256(pubkey)[:16], appended to a
  genesis-rooted chain, Article-12 log line, harness self-hash + toolchain pin + lake-manifest hash
  in every receipt. `szl_geh_verify.py` needs only the standard library and **recomputes the gate
  from evidence** rather than trusting the `decision` field.

Sandbox evidence (throwaway key `211daec220c02193`, pub `b755d706…435d`):

| Run | Verdict | Receipt sha256 |
|---|---|---|
| `--prove` (REPL) | KERNEL_VERIFIED_AND_SIGNED / PASS | `aa173bb9…4ff24` |
| `--bench-crypto 10` (REPL) 10/10, contamination clean | PASS | `3523acd9…fb171` |
| `--bench-crypto 4` (SIMULATED) | TRACE_SIGNED_KERNEL_UNVERIFIED / VOID | `4a3fa783…f065d` |
| `--audit-file SelfTest.lean` | PASS | `c376e58a…52170` |
| `--audit-file Red_NativeDecide.lean` | VOID (`Lean.ofReduceBool`) | `8949ad02…a3132` |

Chain head `bdb60e93…0890`; `szl_geh_verify.py --evidence …` → 5 receipts, 0 failures; pytest
11 passed; harness sha256 `475c1643…ab63`; guard sha256 `6ed9fb91…2709`.

### A.6 Added after the first pass (2026-09-30 evening)

* **Hub vs. local control:** `SZLHOLDINGS/chaski-r2` (modified 2026-09-30 21:54Z) publishes
  `adapter_model.safetensors` with LFS sha256 `6f12981e…cdde6` (25,587,104 bytes). The local r2
  control's weights hash recorded in the thread was `a16be6dd…ea17` — same size, different bytes.
  Either a re-serialization or different weights; the runbook now reports
  `r2_local_matches_hub_published` so you can tell which. Until then, "the Hub r2 is the control"
  is an unverified claim.
* **Silent adapter no-op hazard (found by the probe, mechanism verified):** the published
  `SZLHOLDINGS/chaski-r2` adapter is keyed for the multimodal module layout
  (`base_model.model.model.language_model.layers.*`). Under transformers 5.18,
  `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")` returns `Qwen3_5ForCausalLM`
  (`model_type qwen3_5_text`, modules `model.layers.*`); PEFT then applies **0 of 192** adapter
  tensors and only emits a `UserWarning`. In that configuration the "chaski-r2" candidate
  reproduced the committed **base-model** outputs byte-for-byte on all 11 held-out cases
  (sha256-equal to the `base-qwen35-0.8b` rows of receipts A and B: 0/5 drafts, 6/6 five-token
  `REFUSE`). Loading the same adapter through `AutoModelForImageTextToText`
  (`Qwen3_5ForConditionalGeneration`, modules `model.language_model.layers.*`) applies 192/192 and
  yields the seven-key compact drafts that are the r2 signature. Two consequences: (1) the pristine
  canonical runner uses `AutoModelForCausalLM`, so which class it instantiates — and therefore
  whether any adapter is applied at all — depends on the installed transformers version, and a
  no-op adapter would be recorded as a MEASURED 0/5 with no error; (2) anyone following the
  standard PEFT recipe on current transformers evaluates the base model and believes it is
  chaski-r2. `tools/chaski_margin_probe.py` now fails closed (`ADAPTER_NOT_APPLIED`) unless every
  checkpoint tensor lands, and records `adapter_keys` and `loader_class` in the receipt. The same
  assertion belongs in the canonical runner and in every model card's loading snippet.
* **Cross-environment determinism datapoint:** CPU fp32 + transformers 5.18 reproduced the
  owner-metal CUDA bf16 base-model outputs byte-for-byte on 11/11 held-out cases. The base model
  is not environment-sensitive on these prompts; whatever flips between receipts A and B is
  adapter-side.
* **Control reproduced off-metal:** with the correct loader class, the Hub `chaski-r2` adapter
  scores **5/5 drafts and 6/6 refusals on CPU fp32** (transformers 5.18, torch 2.14) through the
  pristine runner's own prompt path and scorers — the r2 control is reproducible outside the owner
  machine. Its structural decision after `"executed":false` is `,"` at +7.2 logits over the
  runner-up (`}` not in the top 3): no knife edge. The free-text `claim` field has the expected
  close word choices (min margins 0.01–0.22), which is why the flag is now restricted to structural
  `}`-vs-`,` decisions. Receipts: `evidence_sandbox/chaski_probe/`.
* **Margin probe shipped:** `tools/chaski_margin_probe.py` replays the held-out prompts through
  the pristine runner's own prompt path and scorers and records, per case, the top-2 logit margin
  at every generated token, the environment (torch/transformers/peft versions, model class,
  `config.architectures`, dtype, chat-template hash) and whether generation ended on EOS. It writes
  a DIAGNOSTIC receipt, never the canonical one. Smoke-tested on CPU with a tiny model and a LoRA
  adapter; the base-model CPU replay result is recorded in Part C when available.

* **Coverage preflight everywhere the gate can run:** `chaski_margin_probe.py --coverage-only`
  loads base + adapter with the chosen loader class, verifies every checkpoint tensor landed and
  exits 4 otherwise (verified live: text class 0/192 → exit 4; multimodal class 192/192 → exit 0).
  The runbook's `-RunCanonicalGate` path now runs this preflight with the pristine runner's own
  class (`AutoModelForCausalLM`) for every adapter and refuses to start the gate on failure — the
  pristine runner itself stays byte-identical (`f4ca282a…`).
* **Canonical runner hardened (separate branch `chaski/adapter-applied-guard`):**
  `chaski/adapter_guard.py` + wiring in `chaski/bakeoff_named_n.py::load_runtime` raise
  `AdapterNotApplied` unless all checkpoint tensors land, and each candidate row gains a `loader`
  field (`wrapper_class`, `model_class`, `model_type`, `transformers`, `peft`, `adapter_keys`).
  Six pure-Python tests; verified against the real Hub adapter on CPU (192/192 vs raise).
* **lutar-lean locked set, kernel-checked here:** the 24 theorems carrying the locked formulas
  {F1,F4,F7,F11,F12,F18,F19,F22} in `Lutar/Puriq/Formulas/ProvedFormulas.lean` @ main
  `c4af7e22` pass `#geh_guard` under a bare Lean v4.18.0 core build (no Mathlib): 11 depend on no
  axiom, 10 on `propext` only, 3 on `propext` + `Quot.sound`, 0 on `Classical.choice`; a red
  control (`sorry`) is refused with `sorryAx`. Shipped as `tools/lean/LockedGuard.lean` + a
  lake-build.yml step asserting exactly 24 `allowed:true` lines (branch
  `ci/locked-8-axiom-guard`).
* **lutar-lean drift found on the way:** `PuriqFormulaLean.lean` is listed on PROVEN_FORMULAS.md
  as a source for the locked 8 but is **not in the build graph** (`Lutar.lean` imports only
  `ProvedFormulas`; the file's own note says the still-open formulas live there), and it does not
  compile standalone under v4.18.0: `f6_lmdb_durability` fails at 456:2 (`simp made no progress`)
  and F23 is a labeled `sorry` (Conjecture 1). Its footer records a bare Lean 4.13.0 compile, so
  this is toolchain drift on an unbuilt file, not a false locked-set claim — but the "Source" link
  for the locked 8 should point at `ProvedFormulas.lean` alone.

## Part D — Limits, stated plainly

* **Pantograph toolchain mismatch:** PyPantograph 0.3.15 pins `leanprover/Pantograph` @ `842c0fe6`,
  whose `lean-toolchain` is `leanprover/lean4:v4.29.1`. It cannot import a v4.18.0 `geh-lean`
  build. A Pantograph lane therefore needs either a second guard build under 4.29.1 or an older
  Pantograph pinned to 4.18 — a separate, bounded task, not a flag flip.

* Pantograph and LeanDojo transports are API-corrected but **not kernel-exercised** in this build;
  their receipts carry `transport_verified_here=false` and the verifier treats only evidence, not
  names. The REPL is the exercised kernel path.
* `HeuristicProver` is a deterministic stand-in. Wire your model through `CallableProposer` before
  quoting any pass@k as a model result.
* The Windows runbook was reviewed but not executed here (no PowerShell in the sandbox). It is
  read-only except for the stated paths and stops nonzero on the first failed gate.
* The sandbox signing key is throwaway; your owner key is generated on first run under
  `chaski_r4\evidence\geh\geh_keys\` — back it up, never commit the `.pem`.

## Part E — Next bounded actions

1. Run `geh_v8_runbook.ps1 -Bundle …` (no gate). Read the reconciliation block: it tells you which
   r4 adapter is on disk.
2. Run it again with `-RunCanonicalGate`. Promote the new receipt to the canonical path only by an
   explicit copy after reading it; then close the thread's open item in the ledger.
3. Drop `#geh_guard` into `lutar-lean` CI for the locked theorem set {F1,F4,F7,F11,F12,F18,F19,F22} — it converts `SORRIES.md`
   discipline from a script that counts to a build that fails.
4. Add `transformers`/`peft` versions, model class, rendered prompt and top-2 logit margin per case
   to the chaski runner receipt; that is the receipt upgrade this whole thread was asking for.
5. Replace `HeuristicProver` with the chaski adapter via `CallableProposer` and publish the first
   randomized-syntax prover benchmark receipt with an honest `research-only` tag.

Sources: [szl-forge canonical receipt](https://github.com/szl-holdings/szl-forge/blob/main/chaski_r4/bakeoff_canonical_four_way_r4.receipt.json) · [szl-forge four-way receipt](https://github.com/szl-holdings/szl-forge/blob/main/chaski_r4/bakeoff_four_way_receipt.json) · [training_receipt.json](https://github.com/szl-holdings/szl-forge/blob/main/chaski_r4/training_receipt.json) · [training_plan.json](https://github.com/szl-holdings/szl-forge/blob/main/chaski_r4/training_plan.json) · [commit bcdd1d85](https://github.com/szl-holdings/szl-forge/commit/bcdd1d85) · [lutar-lean lean-toolchain](https://github.com/szl-holdings/lutar-lean/blob/main/lean-toolchain) · [Qwen3.5-0.8B chat_template.jinja](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja) · [Lean REPL](https://github.com/leanprover-community/repl) · [PyPantograph server.py](https://github.com/stanford-centaur/PyPantograph/blob/main/pantograph/server.py) · [DSSE protocol](https://github.com/secure-systems-lab/dsse/blob/master/protocol.md)
