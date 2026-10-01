# GEH v8 — SZL Governed Evaluation Harness (Lean 4 lane)

Fail-closed, local-only harness that turns theorem-proving into governed, hash-linked tool-calls
against a real Lean 4 kernel (leanprover-community/repl @ v4.18.0 — the lutar-lean pin), audits
every finished proof's axioms with a build-failing guard (`#geh_guard`), and signs the result as a
DSSE proof receipt in an append-only chain. Verified end-to-end before shipping (see THREAD_AUDIT.md).

## Layout
```
szl_geh_v8.py            harness (CLI + library)
szl_geh_verify.py        zero-dependency offline verifier (pure-Python ed25519, recomputes the gate)
chaski_margin_probe.py   DIAGNOSTIC receipt upgrade for the chaski named-N bake-off (per-token logit margins + environment)
geh_v8_runbook.ps1       Windows one-paste runbook (PowerShell 5.1)
tests/test_geh_v8.py     pytest (structural + kernel tests)
geh-lean/                Lake project: GEH/Compliance/GehGuard.lean, SelfTest.lean, RedTests/, lakefile.toml (requires REPL @ v4.18.0)
evidence_sandbox/        receipts + chain produced in the build sandbox (throwaway key; public key only)
THREAD_AUDIT.md          audit of the source thread (chaski r4 + GEH v3–v7) and v8 evidence
GEH_v8_CODEX_PAYLOAD.md  single-file task payload for Codex (all sources inline)
```

## Quick start (Linux/macOS)
```bash
pip install cryptography pytest
cd geh-lean && lake update && lake build && lake build repl && cd ..
python szl_geh_v8.py --root . --lean-project ./geh-lean --selfcheck
python szl_geh_v8.py --root . --lean-project ./geh-lean --transport repl --bench-crypto 10
python szl_geh_v8.py --root . --lean-project ./geh-lean --verify-chain
python szl_geh_verify.py --evidence ./chaski_r4/evidence/geh
python -m pytest -q tests/
```

## Windows (owner metal)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Downloads\geh_v8_runbook.ps1" -Bundle "$env:USERPROFILE\Downloads\geh_v8_bundle.zip"
```
Add `-RunCanonicalGate` to also run the chaski canonical four-way bakeoff into a NEW receipt.

## Doctrine
sorry/admit/native_decide/foreign axioms VOID a receipt (gate + kernel audit); plausible/slim_check
are EVIDENCE_ONLY; SIMULATED never certifies; PASS needs live no-goals AND audit 0 errors/0 sorries
AND `#geh_guard allowed=true` AND artifact grade COMPLIANCE_CANDIDATE. No network, training, Hub or
Git writes. EU AI Act Art. 12 fields are an engineering mapping, not legal advice.
