# Runbook executed under Windows PowerShell 5.1 (GitHub-hosted windows-latest, run 36803274253)

Workflow `geh-v8-windows-preflight.yml`, 2026-10-01. `$PSVersionTable.PSVersion` = 5.1.26100.33438.
Simulated owner root staged from a CRLF git checkout; elan v4.2.3 + Lean 4.18.0 installed on the
runner; `lake update` / `lake build` / `lake build repl` from scratch on Windows.

| stage | result |
|---|---|
| manifest (LF-normalized) | 14/14 ok on CRLF files |
| red tests | Red_Axiom, Red_NativeDecide, Red_Sorry all correctly failed the build |
| self-check, tool-use spec | pass |
| kernel proof receipt | PASS, sha256 `56994dc2fbc666559c1438175aa300a7bf94dd581fb051afbcdb4464390f6804` |
| K=10 kernel benchmark receipt | PASS, sha256 `dc03832defebe21f9239c95df49499eed2f2ec5ec62db2d0fe5f9f9e3727defe` |
| chain | 2 receipts, head `dc03832d…`, verified on the runner by both verifiers and again off-runner by `szl_geh_verify.py` (signatures valid, gates recomputed PASS) |
| runbook receipt | `geh_v8_runbook_20261001_015508.json`, exit 0, `COMPLETE` |

Defect found only by running real 5.1: `Set-Content -Encoding UTF8` writes a UTF-8 BOM, so the
runbook receipt did not parse with `json.load`. Fixed in the runbook (BOM-free `WriteAllText`);
the receipt in this folder is the BOM-prefixed original from the run, kept as evidence.
The signing key was generated on the runner and discarded; only the public key is kept here.
Preflight only: no GPU, no local adapters, `-RunCanonicalGate` not passed. Proves nothing about a model.
