# szl_evolve: provenance toolkit (stdlib; PyYAML for card_lint; mlcroissant optional in tests)

| Module | Purpose | Leader pattern it follows |
|---|---|---|
| `run_manifest.py` | Pinned training-run manifest + fail-closed validator (base/data/code 40-hex pins, sha256, cost cap, private output, contamination verdict, license-laundering rule) | SLSA: training is a build, data/models are dependencies; OpenSSF OMS manifests |
| `decon_ext.py` | Sampled n-gram + cluster-expansion detection, MinHash/LSH near-dup, group-disjoint and cross-campaign leak matrix | allenai/decon (Olmo 3); NeMo Curator fuzzy dedup; lm-eval 13-gram decontamination |
| `croissant_rai.py` | Croissant 1.0 + RAI 1.0 JSON-LD from a JSONL file; RAI text only from supplied metadata | MLCommons Croissant / Croissant RAI |
| `preupload_check.py` | Pickle import scan (no unpickling), SSN/card/email/secret patterns, private-by-default rule | HF pickle/malware/secret scanning, applied before upload |
| `card_lint.py` | Card linter CL001-CL040 | HF card metadata docs; Croissant RAI fields |
| `advisory_review.py` | Advisory-only second opinion from a local Ollama model on card_lint findings and near-dup pairs. Receipt records model, digest, prompt/response sha256. Never a gate: lint and near-dup results pass through unchanged; fails open if the model is unreachable; refuses `:cloud`/`-cloud` tags and digest mismatch | Local review only (kimi-estate-ops digest prefix ca1b2de39499) |

Run: `python -m pytest tests/evolve -q` from the repo root. Nothing here uploads, trains, signs or publishes.
A VALID manifest or a CLEAN gate never authorizes a launch.
The advisory reviewer cannot change a verdict or a lint error count.
