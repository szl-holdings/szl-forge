# Provenance of `ops-health/`

**SYNTHETIC.** This page is generated together with `PROVENANCE.json`, which is the machine-checked
form. `tools/verify_provenance.py` checks every statement below from the landed bytes; the CI
workflow `.github/workflows/oac-ops-health.yml` runs it.

## Source

- **Research repository:** szl-work/oac-frontier (local git repository; unpublished). Its commits are cited by full sha;
  they cannot be fetched from here.
- **Copied from research revision** `261ae5f0383d9fb56d2192ac7be74f63439974e7`, the newest research commit that
  changed any copied file. Each file also names the last research commit that changed it and
  its research git blob.
- **Receipt source commit:** `92872f88193242b3fc0c1f604321d889d6988358`. The frozen receipts' source.commit names a commit of the unpublished research repository. freeze.verify_origin, which resolves it with git, is UNAVAILABLE outside that repository; the seven tests that need it skip here with that reason.
- **Line endings:** every file is LF-only; the logs whose research copies carry CR bytes (transform eol_lf) were converted CRLF -> LF and lone CR -> LF; none of them is hash-bound. research_sha256 is the research copy's digest.
- **Not committed:** the registered data splits and the AM-4 known-good train splits. `tools/regenerate_data.py` rebuilds each byte-exact:

| path | record | sha256 |
| --- | --- | --- |
| `v2/data/calibration.jsonl` | split:calibration | `526d47c5e922aa35e3fba1f46f595fad6cc0fd4b3fd2e52114b5c99247e57b16` |
| `v2/data/conformal.jsonl` | split:conformal | `bf317275e207e13ddd38d95b308eb7ff6a21d13e71b27622d8e65f4a4eb5f2dd` |
| `v2/data/sealed/shift_clock_skew.jsonl` | split:shift_clock_skew | `d98e19e8fd1e9d3c5e8c2c1c808492d8c0f6be378ef0d89cc50569afb9c74823` |
| `v2/data/sealed/shift_failure_bursts.jsonl` | split:shift_failure_bursts | `4199085505a0e3ea3ead9b282eb28d4c8c0060a3034fdee713eb845ee97a8c0e` |
| `v2/data/sealed/shift_missing_tls.jsonl` | split:shift_missing_tls | `a4a96189c864c93906649ab629f3ed979b395b9252df8dc5bb777fc8acf0bedc` |
| `v2/data/sealed/shift_queue_saturation.jsonl` | split:shift_queue_saturation | `c647bb89be02205fa8961a9ed3209c1498cc0f7524224365c2af6596baae85da` |
| `v2/data/sealed/test.jsonl` | split:test | `6479d0bffab20771ba1f5976f95dec8825e8edf0aa4e4faa559d77d57fec1f3b` |
| `v2/data/train.jsonl` | split:train | `5211e7ec954ec00c743bd62dae107f7a80335e4e5962a31b788e7bf56ecf0adb` |
| `v2/data/validation.jsonl` | split:validation | `d0a37356784dcd60a4dbea68d760c55d56a7bd69b3a6f4dc833cfdbb38a951f3` |
| `v2/results/known_good/k1/train.jsonl` | known_good:k1 | `68860affab3b74d8e19cbb4a88cb5ea3a862f3a7e5342ce05581f88e95134d4e` |
| `v2/results/known_good/k2/train.jsonl` | known_good:k2 | `1be3d6898edac4ab412b5c2675543006a79e83132aae3382ba7b7b78105913df` |
| `v2/results/known_good/k3/train.jsonl` | known_good:k3 | `a21ddeff822a6be1e34a071678b68db84d6eb27d06dc5dfb23a1b641472c5725` |
| `v2/results/known_good/k4/train.jsonl` | known_good:k4 | `49630c78c510f31d8838eef57970990150d257bd9f9cf431cee0e50c0dfa95d8` |
| `v2/results/known_good/k5/train.jsonl` | known_good:k5 | `33744d8e6cae2bff7262b3f2a5b963280febd5fec5086a84e84178e75883b477` |

## Bound hashes

182 distinct SHA-256 values are named by the landed files under `v2/` and `runs/`
(the `.jsonl` row files are bound through their manifests). Each has exactly one rule:

| rule | values | meaning |
| --- | --- | --- |
| cited_blob | 2 | a research-repository file; declared, not checkable here |
| cited_scratch | 1 | a git-ignored research scratch file; declared only |
| crlf_roundtrip_equal | 1 | the landed LF copy, with LF turned into CRLF, hashes to it |
| equal | 73 | a landed file hashes to it |
| forge_blob | 1 | a git blob of this repository |
| in_cited_blob | 1 | a field of such a blob |
| receipt_chain | 43 | inside the governed search receipts, whose chain verifies |
| recomputed | 42 | a digest the landed code recomputes (mutants, fail-open fixture) |
| regenerated | 15 | the committed record of a split that `tools/regenerate_data.py` rebuilds |
| v1_hub | 3 | one of v1's published files, staged in this repository |

Values that cannot be checked from this repository (declared, with their research citation):

| value | rule | research path | research commit | reason |
| --- | --- | --- | --- | --- |
| `2765aa14d36869a7…` | cited_scratch | `scratch/v2_final_recheck/final_evaluation.json` | none (git-ignored) | the re-analysis output, written to a git-ignored scratch directory by logs/v2/final_reanalyze_check.py (log logs/v2/final_reanalyze_check.txt); final_recheck.json records its digest |
| `a53252460d8210ca…` | crlf_roundtrip_equal | `v2/research/final_analyze.py` | `cbac49793dc6` | the analysis-time script, committed with CRLF line endings; final_evaluation.json and final_recheck.json bind it |
| `b4dd3922f865ead1…` | cited_blob | `v2/research/mutation.py` | `bb8d56f5d73c` | mutation.py as it was at the recorded mutation run (mutation_readiness.json protocol.git_head); the landed v2/research/mutation.py is a later version |
| `dd0e5acdff504490…` | cited_blob | `CONTRACT.md` | `bca92259eca8` | the research workspace's operating contract at registration (PREREGISTRATION.receipt.json contract_sha256); not published |

## Files

200 files came from the research repository. `PROVENANCE.json` has each one's sha256,
size, research commit and research blob. By last research commit:

| research commit | files | transforms |
| --- | --- | --- |
| `92872f881932` | 30 | eol_lf, none |
| `974159cec1b9` | 20 | eol_lf, none |
| `969f561fceaf` | 18 | none |
| `dff5ae557658` | 12 | eol_lf, none |
| `b64a4e2f15ae` | 11 | none |
| `3b0912a8ae7d` | 10 | eol_lf, none |
| `c46a42c6855d` | 9 | eol_lf, none |
| `ec7874b0596f` | 9 | eol_lf, none |
| `7c7226a4091b` | 7 | eol_lf, none |
| `00e9d1923f83` | 6 | eol_lf, none |
| `852381bf7a96` | 6 | eol_lf, none |
| `db7a5b40b38e` | 6 | eol_lf, none |
| `1b80856c3370` | 5 | none |
| `5ff18027bf78` | 5 | none |
| `8c9ecb3d2b8c` | 5 | eol_lf, none |
| `bd47a2acc48b` | 5 | eol_lf, none |
| `efde7ff3851e` | 5 | eol_lf, none |
| `261ae5f0383d` | 4 | none |
| `d9c5fb3a403a` | 4 | none |
| `946db23ac8d9` | 3 | eol_lf, none |
| `d07b314173e1` | 3 | eol_lf, none |
| `064481f0d603` | 2 | eol_lf |
| `6102fedd71aa` | 2 | none |
| `b75785c1d6c9` | 2 | none |
| `bca92259eca8` | 2 | none |
| `c7e7e5a38fc8` | 2 | none |
| `2ad9b171618b` | 1 | none |
| `7a3ea7c655a2` | 1 | none |
| `929677d9f382` | 1 | eol_lf |
| `aba1bad24883` | 1 | none |
| `cbac49793dc6` | 1 | crlf_to_lf |
| `ed3fc88573a5` | 1 | none |
| `f9c55882c52f` | 1 | none |

Files added for szl-forge and not copied from research: `.gitattributes`, `.gitignore`, `PROVENANCE.json`, `PROVENANCE.md`, `README.md`, `tools/materialize_v1_hub.py`, `tools/regenerate_data.py`, `tools/run_v2_tests.py`, `tools/verify_provenance.py`.
