# Offline CI evidence intake

Executable synthetic CI intake contracts recovered from the owner's mission
handoff. The original normalizer is preserved byte for byte. This package makes
the prototype reproducible in Forge; it does not admit real provider evidence,
invoke the production evidence kernel, or grant execution or release authority.

## Run

From the repository root, with Python 3.10 or newer and no dependencies:

```sh
python -I -B -S tools/ci-intake/run_checks.py > ci-intake-checks.json
```

The runner executes all unit tests, requires at least 53 tests and zero skips,
then checks the sample CLI's expected exit 2 and byte-identical output. It exits
0 only when those local contracts pass. A nonzero sample exit no longer prevents
the remaining checks from running. The report records source hashes and keeps
execution and release permissions false. Standard output is JSON; test details
go to standard error. No provider requests, model calls, subprocesses, installs,
credentials, or GPU jobs are used.

The standalone advisory CLI remains available:

```sh
python -B -S tools/ci-intake/offline_ci_intake.py < tools/ci-intake/sample_input.json
```

Its exit codes are 0 (`READY_FOR_REVIEW`), 2 (`REQUIRES_REVIEW`), and 3
(`INSUFFICIENT_EVIDENCE`). Exit 0 is an advisory label, never permission to merge,
publish, execute, or deploy. The sample deliberately returns 2: a failed attempt,
a later passing retry, and an absent required artifact must remain reviewable.

## Evidence boundary

Policy is a separate caller-reviewed input. Revisions must be exact lowercase
40- or 64-character hexadecimal values at the policy, intake, and result
boundaries. The added revision matrix covers lengths, case, symbolic refs,
nonhex characters, Unicode lookalikes, whitespace, nonstrings and each boundary.

The original 45 tests retain missing/ambiguous artifacts, freshness, failed
retries, missing attempt history, seven distinct test statuses, coverage gaps,
strict input parsing, narrative exclusion, deterministic output and permissions.
Candidate observations remain `DERIVED`, with `NOT_ASSERTED` authenticity,
`NOT_CHECKED` availability, and `UNVERIFIED_INPUT_ASSERTIONS` trust.

Live collection, provider/JUnit adapters, protected policy authentication,
artifact-content verification, rights admission for real data, and integration
with the production evidence kernel are not implemented. Real results must not
be relabeled as synthetic to pass this prototype's admission check.

## Recovery provenance

Recovered from `SZL_CLAUDE_BUILD_TRAINING_EVIDENCE_MISSION (5).md`, 426,103 bytes,
SHA-256 `69d47427a38cbc77ac1c1da9def921916a933b5a265e167a045463e3cd515064`.

| Original file | SHA-256 |
| --- | --- |
| offline_ci_intake.py | 37143f6fc87d5be32b370f22a2822161dc5a98e89b3088e7af8746010e0678c1 |
| test_ci_intake.py | cb097423b67125717088f3371d4d0bfd9d8736171f88df58d99b917d47d9b639 |
| sample_input.json | 13ad005af6d519a86f4aa048aef17dd7f1638385c22c3c59ea9072f8f2d79ebf |
| sample_result.json | 068141c3d027b75f4ddd3c871180eda95111fd19f61f3e2a6635f919604dd351 |

These hashes bind the recovered files, not a signature or independent test
provenance. The obsolete R1 merge checker is not carried forward; a11oy #2944
was verified merged on 2026-10-23.
