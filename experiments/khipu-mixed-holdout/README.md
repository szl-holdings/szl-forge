# Khipu mixed held-out diagnostic: preserved failed record

**Historical result: MEASURED LOCAL DIAGNOSTIC FAILURE.** The recorded original merged checkpoint produced **3/6 NAVIGATE, 0/6 ABSTAIN, 7/12 contract-valid** on 12 constructed fictional cases. All six abstention cases proposed NAVIGATE; two navigation outputs were invalid JSON and one failed the schema.

The current [run.py](run.py) verifies recorded bytes and independently regrades saved outputs offline. It performs no inference, downloads, training, controller action, provider publication, or evidence writes. It never imports or executes [the archived original runner](evidence/frozen-runner.py.txt). A consistent failed record is not a passing model gate.

## Preserved bytes

| Artifact | SHA-256 |
|---|---|
| [Original runner, archived as text](evidence/frozen-runner.py.txt) | `c2498ec5cdfb63ca625f0a0af52eaed995231d6874711b25b76d55916499504c` |
| [Original schema](evidence/plan-schema.json) | `b95f9927366dae7c5d36cfb7de6e229eb605524318ab642a6aa2292a212170d0` |
| [Cases](cases.json) | `ab19127b68f089e379457e3ea70c14a4c88217136b18fadb1f994542b6ad75dc` |
| [Preregistration](preregister.json) | `33781211baf1e4fa4446593eb0f0766fa6ece2725c9cfa68729f0fa15b824b5d` |
| [Repository LF receipt](evidence/receipt.json) | `e3e6ce706c0afe001146e156682e75917fdd9367ef707b69a5835ecbd3125d86` |
| [All 12 raw outputs](evidence/outputs.jsonl) | `190f18728ac0d0cf7401ed2f4553176ffc60b1681abbbe1697c3f499a0f92313` |

Cases, preregistration, recorded receipt and output bytes have not changed. The original Windows CRLF receipt serialization has SHA-256 `bf8288766c7eee231cd4d97481bd3067b4de800b15b175171b9b4634cd133e10`; the repository's LF copy is listed above. The verifier pins that repository copy rather than silently claiming both serializations have the same digest.

The runner archive retains every original byte, including its unused import that failed active Python lint. It is now historical data, not executable code or a CI exclusion. Active Python source passes the unchanged Ruff gate. Original source, environment notes, replay procedure, limitations and failed history remain available at [the historical PR head 7752de17](https://github.com/szl-holdings/szl-forge/tree/7752de17c71de4130d9b42df4ab38ceb02f96cef/experiments/khipu-mixed-holdout).

## Offline verification

From the repository root with its test dependencies installed:

```bash
python experiments/khipu-mixed-holdout/run.py --verify-recorded
python -m pytest -q tests/test_khipu_mixed_archive.py
```

The verifier hashes and reads each preserved artifact once, validates the historical receipt bindings, requires all 12 cases, independently recomputes every selected-field grade, and compares aggregate metrics with the original receipt. Its stricter JSON parser rejects duplicate keys and non-JSON constants. None of the 10 parseable saved outputs has duplicate keys, so the original grades remain unchanged; the two invalid JSON outputs still fail. Any byte, grade, case coverage, or metric mismatch refuses verification.

The old inference options are deliberately unavailable. A future model replay requires a separately preregistered runner, qualified local assets/environment, a new run and a new receipt. The original runner's Windows working-tree bindings for `publishing/model-source-bindings.json` and `khipu/manifest.json` used CRLF although their Git blobs are LF. This archive verification does not claim those historical model assets or that environment have been independently restored.

## Evidence limits

- The recorded Forge source base is `f4688d2233ed4123995fab7861b9a074f60cc3d5`; the immutable original Hub revision is `8a5c666c4fc1ec75a51b8bd9784b4504e807757a`. These are historical subjects, not qualification of current source.
- The fixture is SIMULATED, fictional, outside the canonical curriculum and `training_eligible: false`. Exact query/handle exclusion does not prove absence of semantic leakage. Published cases cannot be a private future promotion set.
- The public owner-reported **2/6** on `khipu/adversarial.jsonl` is a separate denominator and remains unchanged. Do not pool it with these cases.
- The preregistration timestamp is self-reported. Its first signed Git anchor came **after** inference. Matching hashes do not establish an independently witnessed preregistration.
- The receipt is unsigned. Independent model inference replay is **NOT_RUN**. Original adapter coverage was structural, not numerical equivalence; actual immutable training-implementation revision is UNKNOWN.
- `7/12 contract-valid` concerns structure and selected fields, not free-text truth or a functioning controller. ABS-01 included a false content-resolution assertion despite structural validity. Its case label still fails. No controller/tool invocation occurred.
- The historical source describes a local RTX 5050 Laptop GPU run, greedy seed 142857, maximum 640 new tokens, the pinned merged checkpoint, and separate adapter inspection without reapplication. The historical environment/tokenizer and disk-capacity caveats remain recorded at the immutable source link.
- Joules are UNAVAILABLE and carbon is ROADMAP. Model promotion remains false; Khipu proposes JSON and A11oy/Alloy remains the decision boundary.
