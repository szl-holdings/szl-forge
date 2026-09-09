# Payload 1 — SZL Hugging Face Frontier Backend v2

Repository: `szl-holdings/szl-forge`.
Authority order: **GitHub → Hugging Face → a-11-oy.com → a11oy.net**.
Scope: public candidate metadata, evidence integrity, repeatable observations and
non-serving consumer projections. This is not model qualification or deployment.

## Continue the existing implementation, not another disconnected build

PR #202 merged the original lane as `99d73a290de79e65e76bafb69a7ffa84fbc8c76c`.
This upgrade is based on inspected main `38d1184473e2928ee06a8cb573dee0f5cd088eef`.
The v1 implementation allowed explicit HOLD/REJECT to fall through to EVALUATE,
accepted absent gating as clear, selected the first license declaration, accepted
ambiguous JSON, and had no receipt verifier. v2 closes those gaps.

Files: `inference/hf_frontier.py`, existing `tests/test_hf_frontier.py`, new
`tests/test_hf_frontier_v2.py`, `.github/workflows/hf-frontier-observation.yml`,
and this single handoff document. The runtime package remains dependency-free.

## What v2 implements

Explicit HOLD and REJECT remain absorbing; WATCH remains research-only even with
blockers. Missing or malformed access-gating information does not imply clearance.
Conflicting card/tag licenses become HOLD rather than selecting a convenient term.
Repository identity fields must agree and references must be immutable SHA pins.
Remote-code signals include metadata auto_map fields and repository Python files,
not just custom_code tags. These are observations, not an exhaustive source audit.

The bounded JSON reader rejects duplicate keys, NaN, Infinity, exponent overflow,
invalid Unicode, non-object roots and oversized bodies. The public client refuses
redirects, has finite timeouts, and makes at most three GET attempts per candidate.
It honors bounded numeric Retry-After values; longer/date-form delays are deferred
as failures, never converted to permission. No auth header or model loader is used.

Refresh isolates individual upstream failures into explicit PARTIAL/UNAVAILABLE
reports while retaining valid observations for other candidates. It never reuses
stale success to hide a current failure. CLI exit 2 means incomplete observation;
exit 1 means invalid evidence or an operational error; exit 0 is complete metadata
observation or successful offline verification, not model approval.

Every observation has a receipt hash. The manifest verifier checks nested hashes,
the trusted registry, complete/unique coverage, normalized fields, recomputed gates,
source identity, and authority flags. A separate semantic digest excludes scan time
and popularity counters so repeated observations do not manufacture frontier events.

Projection generation requires a source revision, verifies the manifest, and rejects
observations older than 24 hours or more than 60 seconds in the future. It labels
model evaluation and runtime verification NOT_PERFORMED. Its production disposition
is HOLD and all execution/publication/promotion flags remain false.

Important: hashes provide integrity checks, NOT authenticated provenance. An attacker
who controls all source evidence can recompute hashes. Source trust must be anchored
by the protected GitHub commit and the exact trusted workflow run/artifact. Atomic
local writes are not immutable storage. Artifact retention is 30 days, not permanent.

## Candidates and independent evaluation lanes

`bodhan-ai/indic-transcribe-core`: multilingual/code-mixed ASR. On the connected
Hub inspection of 2026-09-09, it was gated, license other, and custom_code-tagged.
Keep HOLD pending actual license/access/source review. Later measure WER/CER per
language, language-ID errors, code-mixing, p50/p95 latency, RAM/VRAM and realtime
factor on consented, rights-cleared independent audio. Preserve failed cases.

`bodhan-ai/indic-speak`: multilingual/code-mixed TTS. The same inspection showed
gated access and license other. Keep HOLD. Later measure intelligibility,
pronunciation, consistency, time-to-first-audio, realtime factor and resource cost;
require consent for any real person's voice and explicit abuse/provenance controls.

`gdiamos/amx-reasoning-v1-instruct`: CPU/kernel architecture WATCH only. Its upstream
card describes a custom architecture rather than a Transformers drop-in. Observe
transferable CPU/attention mechanics; do not imply that a license tag, checkpoint
size, or upstream score establishes SZL production reasoning quality. Reproduction
requires a reviewed runtime and matching supported hardware in a separate change.

No weights are downloaded, retrained, rehosted, or executed by this payload.

## Execute and verify

Run from the existing Forge repository root with Python 3.11 or later.
The original eight pytest tests remain in base CI; the v2 unittest suite needs no
additional packages. Full-repository validation still belongs to the protected CI.

```bash
python -m unittest discover -s tests -p 'test_hf_frontier_v2.py' -v
python -m pytest -q tests/test_hf_frontier.py tests/test_hf_frontier_v2.py
python -m inference.hf_frontier
```

Explicit live observation on a clean checkout (POSIX shell):

```bash
python -m inference.hf_frontier --refresh \
  --source-revision "$(git rev-parse HEAD)" \
  --output artifacts/hf-frontier/manifest.json \
  --projection-output artifacts/hf-frontier/projection.json
python -m inference.hf_frontier --verify artifacts/hf-frontier/manifest.json \
  --source-revision "$(git rev-parse HEAD)"
```

The source SHA above is an assertion by the operator. For trusted publication,
verify the clean checkout and use the exact protected-main workflow artifact.
Do not stamp a branch, dirty checkout, or stale artifact as a deployed-main witness.

## One Python handoff block (offline by default)

```python
from pathlib import Path
import json
import subprocess
import sys

root = Path.cwd()
paths = ["inference/hf_frontier.py", "tests/test_hf_frontier.py",
         "tests/test_hf_frontier_v2.py",
         ".github/workflows/hf-frontier-observation.yml"]
missing = [name for name in paths if not (root / name).is_file()]
if missing:
    raise SystemExit(f"missing frontier files: {missing}")
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                "-p", "test_hf_frontier_v2.py", "-v"], check=True)
plan = json.loads(subprocess.check_output(
    [sys.executable, "-m", "inference.hf_frontier"], text=True))
for flag in ("weightDownloadAuthorized", "remoteCodeExecutionAuthorized",
             "trainingAuthorized", "publicationAuthorized",
             "deploymentAuthorized", "automaticPromotionAuthorized"):
    if plan.get(flag) is not False:
        raise SystemExit(f"unexpected authority: {flag}")
print("Offline frontier verification passed. No live scan or deployment performed.")
```

## Automation and downstream handoff

The new workflow runs offline contracts on PRs. On protected-main push, manual
main dispatch, and a six-hour schedule, it observes only the three public model
metadata endpoints and stores source-bound manifest/projection artifacts. There is
no Hugging Face token, provider mutation, repo write, merge bot, hardware purchase,
training job, or production rollout in this workflow. An upstream failure leaves
the observation job non-green and retains the available failure evidence.

After this PR passes exact-head checks and review, merge through the normal
protected path. Re-read the final main SHA and the first main observation artifact.
Verify the artifact's run/repository/source binding outside the self-hash verifier.
Consumers must explicitly adopt v2; old v1 evidence is not silently upgraded.

The downstream projection file is a CONTRACT, not proof that sites consumed it.
Use the existing single-writer workflow and source-binding conventions in each
consumer. Open a focused consumer PR in `szl-frontier` only after inspecting its
current renderer/manifest schema; do not overwrite the catalog or create new Spaces.
Expose unavailable/partial/stale separately from a clean metadata scan. Keep actual
runtime eligibility separate on a-11-oy.com; expose receipts and bounds on a11oy.net.
Re-observe deployed source plus semantic/artifact parity before claiming alignment.

Keep `.github#728` inventory-scope drift and `lyte-services#18` exact-source drift
independent. Their current state must be re-read before remediation; this lane does
not resolve them. The existing `szl-frontier#55` ghlore project-memory intake is a
separate candidate: do not duplicate it or ingest private issue/PR corpora here.

## Completion and rollback

Done for this slice means tested source, protected merge, and a real main-branch
metadata observation artifact. Done for the ecosystem additionally requires each
consumer's admitted update and live readback; this source PR does not claim that.
Rollback is a reviewed revert of this change, followed by rechecking consumers.
Do not roll back to v1 evidence as though its weaker verifier were equivalent.

## Primary references

- https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
- https://huggingface.co/bodhan-ai/indic-transcribe-core
- https://huggingface.co/bodhan-ai/indic-speak
- https://huggingface.co/gdiamos/amx-reasoning-v1-instruct
- https://github.com/szl-holdings/szl-forge/pull/202

These sources establish metadata/declared behavior, not independently reproduced
model performance, license approval for SZL's intended use, or live deployment.
