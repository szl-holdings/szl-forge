# szl_evolve: provenance toolkit (stdlib; PyYAML for card_lint; mlcroissant optional in tests)

| Module | Purpose | Leader pattern it follows |
|---|---|---|
| `run_manifest.py` | Pinned training-run manifest + fail-closed validator (base/data/code 40-hex pins, sha256, cost cap, private output, contamination verdict, license-laundering rule) | SLSA: training is a build, data/models are dependencies; OpenSSF OMS manifests |
| `decon_ext.py` | Sampled n-gram + cluster-expansion detection, MinHash/LSH near-dup, group-disjoint and cross-campaign leak matrix | allenai/decon (Olmo 3); NeMo Curator fuzzy dedup; lm-eval 13-gram decontamination |
| `croissant_rai.py` | Croissant 1.0 + RAI 1.0 JSON-LD from a JSONL file; RAI text only from supplied metadata | MLCommons Croissant / Croissant RAI |
| `preupload_check.py` | Pickle import scan (no unpickling), SSN/card/email/secret patterns, private-by-default rule | HF pickle/malware/secret scanning, applied before upload |
| `card_lint.py` | Card linter CL001-CL040 | HF card metadata docs; Croissant RAI fields |

Run: `python -m pytest tests/evolve -q` from the repo root. Nothing here uploads, trains, signs or publishes.
A VALID manifest or a CLEAN gate never authorizes a launch.
