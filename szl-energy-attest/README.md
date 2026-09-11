---
license: apache-2.0
tags:
- szl
- governed-inference
- energy-attestation
- receipts
---

<p align="center">
  <img src="https://raw.githubusercontent.com/szl-holdings/szl-forge/main/szl-energy-attest/card/holo-banner.svg" alt="szl-energy-attest — the attested joule as a hologram: a receipt stub with a gold seal, chained on both sides" width="100%"/>
</p>

# szl-energy-attest

**The cut, in one line:** the joule is typed, chained, and never invented.

Attestable energy receipts for governed compute — MEASURED NVML joules, or an honest `UNAVAILABLE`/`null` when no meter is present. Never fabricates a joule.

**Canonical GitHub source:** https://github.com/szl-holdings/szl-energy-attest (this Hub repo is the publish mirror).

## What it does

- Records measured GPU energy (NVML joules), tokens-per-joule, advisory policy-gate status, and tamper-evident SHA-256 hash-chained receipts for inference runs.
- Absorbs the deprecated `governed-inference-meter` surface (see its Hub card, tagged `deprecated`).
- Energy labels are typed: `MEASURED` (live meter), `REPORTED` (external context), `UNAVAILABLE` (no meter — recorded as null, never invented).

## Doctrine boundary

- Λ (Lambda-Spine) aggregation is **Conjecture 1** — advisory, never a theorem.
- Unsigned receipts are labeled `UNSIGNED` honestly; DSSE/ECDSA-P256 signing rolls in via `szl-receipt`.
- Locked-proven formula count is exactly **8** `{F1,F4,F7,F11,F12,F18,F19,F22}`.

Part of the SZL Holdings governed-AI estate: https://github.com/szl-holdings · product https://a-11-oy.com · proof https://a11oy.net
