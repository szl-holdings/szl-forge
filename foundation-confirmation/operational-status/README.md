# Source-bound operational observation, 3 October 2026

The [machine-readable observation](2026-10-03T070728Z.json) records the protected
source upgrade, public inference witnesses, Windows manual availability, and
remaining qualification gaps. It was assembled on 3 October 2026 at 03:07 EDT
(07:07 UTC). Each surface retains its own observation time in the JSON. Refresh
the live state before making an operational decision.

The measured runtime source is
`3a9ee78ab7cba190d310c69e49c89582e4cc271f`, merged through
[PR #527](https://github.com/szl-holdings/szl-forge/pull/527). This documentation
publishes that historical observation; it does not issue a successor release.

| Surface | Recorded evidence and outcome |
| --- | --- |
| Source and CI | Eleven reviewed operations files matched the protected source. All four jobs in [production run 37101177428](https://github.com/szl-holdings/szl-forge/actions/runs/37101177428) succeeded. |
| Public workbench | Ten hosted files matched. Three actual selector executions had byte-exact POST/GET exports. The [workbench](https://szlholdings-szl-foundation-confirmation.hf.space/) observation was recorded at 01:57 EDT. Continuous availability and independent authenticity were not established. |
| Windows manual availability | One owned task restart, normal priority, all three selectors ready, and a measured 20.766-second startup. The strict audit was recorded at 03:03 EDT. Nine original, three predecessor CPU, and three current CPU trial files remained byte-exact. |
| Automatic recovery | **BLOCKED.** The actual recovery attempt failed its unchanged 420-second observation bound. The retained successor was created 407.333916 seconds after the retry request. The slow phase and precise termination cause remain **UNKNOWN**. The manual restart does not qualify automatic recovery. |
| Scientific qualification | **BLOCKED.** The registered gate remains **FAILED** across 5,184 recomputed evaluation rows. No new training or model promotion was performed. |
| Product and proof origins | The domain observation on 2 October at 23:31 EDT retained 25 exact files and two known Cloudflare HTML-injection mismatches out of 27. Provider-setting repair remained unattempted after the authenticated audit denial. Whole-product qualification remained false. |
| OneDrive | The prior observation retained four byte-checked public release copies. Client synchronization and remote hash readback were unavailable. Cloud backup state remained **UNKNOWN**. |

The JSON's `LOCAL_ARTIFACT_ONLY` publication field describes its state when it
was assembled. Committing these unchanged bytes adds source publication of the
observation. It does not change any of the recorded runtime, recovery,
scientific, domain, synchronization, or authorization outcomes. The existing
[v0.6 release](https://github.com/szl-holdings/szl-forge/releases/tag/foundation-confirmation-v0.6)
is retained; the qualified v0.6.1 successor remains **BLOCKED**.

The evidence catalog contains role, byte count, and SHA-256 references. Private
host paths, owner identifiers, full process arguments, credentials, and private
runtime files are excluded. A digest identifies evidence bytes; this observation
has no digital signature or independent trust authority and does not qualify a
`GovernedAction/v1` proof.

Control before publication: preserve the original 5,655 JSON bytes and SHA-256
`aa9c45feba86f16977be8fe5ea4911b291511ab615ff8dc3e0c9f7be1bd841e7`;
retain `automatic_recovery_verified: false`, the failed scientific gate, and
`qualified_successor_release_issued: false`. JSON parsing, privacy exclusions,
the exact source-copy readback, and `git diff --check` are the local publication
checks. No runtime recovery, provider repair, training, or inference is performed
by this documentation change.
