# Hugging Face evidence consistency audit

The twelve-repository inventory supports repeatable, read-only inspection of the
remediated model cards and their evidence. Each live run resolves a Hub revision
before reading files, retains source hashes and immutable snapshots, and checks
explicit inventory facts against the card. External evidence is pinned to a
declared source revision and expected SHA-256. Repository write permissions,
model qualification, production readiness and inference authority are outside
this audit.

From the repository root, install the existing test extras (`pip install -e
".[test]"`) and run:

```powershell
python -m pytest -q tests/test_audit_hf_estate.py
python tools/audit_hf_estate.py --inventory publishing/evidence-audit-inventory.json --output reports/hf-evidence-audit/audit.md --snapshot-dir reports/hf-evidence-audit/snapshots --fail-on-findings
```

The manual **Hugging Face evidence audit** Actions workflow runs the same live
command on `main`. Before the audit, it checks the existing `HF_ORG_TOKEN`,
`HF_TOKEN`, and `HF_ORG_TOKEN1` secrets in that order using metadata-only HEAD
reads of `MODEL_PROVENANCE.json` at the explicitly recorded Lambda revision.
The first credential with confirmed read access is passed to the auditor's
child process. The other credential variables are removed from that child.
Tokens are never printed, written to reports, or persisted as workflow outputs.
`credential-read-preflight.json` records only the fixed probe identity, attempted
secret names, access results, and selected secret name. This credential probe
does not qualify a model or replace the audit's immutable source snapshots.
If no candidate succeeds, the workflow retains a HOLD access record, still
audits public sources anonymously, and exits nonzero even if those checks pass.
No gate, secret, or repository permission is changed.
Its GitHub permission is `contents: read`; the implementation calls Hub read
APIs only. It does not schedule itself, publish, merge, train or deploy anything.
Reports and replayable snapshots are retained even when findings fail the run.
Only dispatch reviewed source from protected `main`.

For offline replay, repeat the command with `--offline` and the same inventory
and downloaded snapshot directory. Snapshots validate source hashes; edited or
missing source files cannot become a PASS. Keep the inventory used for each run
with its receipt, since its configured checks define the observation scope.

States are `PASS`, `MISSING_BINDING`, `HOLD` and `CONFLICT`. With
`--fail-on-findings`, any non-PASS result exits 1. Without it, a completed report
exits 0 even when the report contains findings. A workflow failure caused by
retained research findings is expected evidence, not permission to weaken the
inventory. PASS is documentation consistency within configured checks only.
The pattern checks are conservative heuristics, not a general truth verifier.

Artifact digests come from Hub LFS metadata unless the artifact itself was
explicitly fetched. A digest appearing somewhere in an evaluation is inadequate:
configured artifact checks require the evaluated filename and SHA-256 together
in the same structured record. Existing historical scores remain historical;
fixing a card or provenance label does not rerun inference or establish a new
held-out qualification.
