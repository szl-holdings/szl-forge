# Energy-Attested Model Card (template)

> Copy into a model's card directory. Fill ONLY from measured receipts
> (`szl.energy-attest/v1`). Never hand-edit a measured value. If a row has no
> receipt, the cell reads UNAVAILABLE — not blank, not estimated.

## Model

- **Repo:** FILL (e.g. SZLHOLDINGS/chaski-r2)
- **Base:** FILL (pinned revision)
- **Merge:** merge_receipt.json (modules applied, weight deltas)

## Measured gates (held-out, on published bytes)

| Gate | Score | Energy (J) | Mean power (W) | Receipt |
|---|---|---|---|---|
| FILL | FILL | FILL or UNAVAILABLE | FILL or UNAVAILABLE | link to receipt json |

## Energy accounting

- **Method:** NVML sampling at 0.1s during gate execution; trapezoidal integration; `szl.energy-attest/v1` receipts, owner-signed into the chain
- **Scope:** GPU package power only. CPU/DRAM/system power UNAVAILABLE unless separately measured
- **Honesty:** every joule on this card traces to a signed receipt. No fabrication, no extrapolation

## Provenance

- Training: DSSE-signed jobspec (szl-gpu-bridge), sha-pinned datasets
- Merge: zero-delta assert passed, receipt attached
- Eval: held-out, run on exact published bytes
- Energy: MEASURED via NVML or UNAVAILABLE — never a claim without a receipt

Doctrine v11. Λ = Conjecture 1 OPEN. Trust ceiling 0.97.
