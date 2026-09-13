# Invariants: independent public byte comparison

## Source and qualification boundary

This read-only verifier complements the receipt inspector introduced by #279.
It is the source admission of the existing four-file local handoff, not another
publisher. Admission starts from Forge commit
`ee7e07c84e5119009ba660f8ba372611c86708af`. Consult the pull request and exact-head
workflow runs for hosted evidence. Offline CI success does not establish a
successful live Hub comparison or publication.

The implementation does not edit `tools/publish_szl_invariants.py`, its release
workflow, credentials, OIDC bindings, source kernel, model weights or existing
receipt inspector. It does not reroute the held publisher integration in #284.

## Prerequisites

Use a clean reviewed Forge checkout and Python 3.11 or 3.12. This verifier has
no third-party dependencies. Obtain three independent full lowercase
40-character commit IDs: GitHub invariants source, Hugging Face model mirror,
and Hugging Face first-class kernel. For release acceptance, the provider pins
must come from the actual authenticated release record. Do not substitute the
GitHub source revision for either provider revision or use mutable `main`/`v1`.

First download the actual release artifact and check its archive SHA-256
against authenticated GitHub artifact metadata. Use the admitted receipt
inspector to bind both reports to the source, publisher, workflow, run and
attempt. Its `EVIDENCE_CONSISTENT_REVIEW_REQUIRED` result is internal
consistency, not independent release proof:

```powershell
python tools/inspect_invariants_release_evidence.py --help
python tools/verify_invariants_hub_bytes.py --help
```

No new successful publication or provider revisions are supplied by this
source change. The historical authorization-only archive must remain rejected.

## Execute the independent comparison

From the reviewed checkout, supply the three independently obtained revisions
in your local terminal. They are identifiers, not tokens:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $Source = (Read-Host 'Exact GitHub invariants source SHA').Trim()
    $Model = (Read-Host 'Exact Hugging Face model SHA from release evidence').Trim()
    $Kernel = (Read-Host 'Exact Hugging Face kernel SHA from release evidence').Trim()
    foreach ($Revision in @($Source, $Model, $Kernel)) {
        if ($Revision -cnotmatch '\A[0-9a-f]{40}\z') {
            throw 'An immutable lowercase revision is required. Nothing observed.'
        }
    }
    $Report = Join-Path $HOME (
        'invariants-public-bytes-' + [guid]::NewGuid().ToString('N') + '.json'
    )
    python tools/verify_invariants_hub_bytes.py `
        --source-revision $Source `
        --model-revision $Model `
        --kernel-revision $Kernel `
        --report $Report `
        --observe
    if ($LASTEXITCODE -ne 0) {
        throw "Byte comparison did not establish equality. Inspect $Report; do not promote."
    }
    Write-Host "Exact file comparison completed. Review $Report with the release provenance."
}
```

Without `--observe`, the program makes no network requests and returns a
non-success status. A local report is still written. There is no automatic
retry, token lookup, upload, model execution or production promotion.

## Closed scope and transport

The verifier reads the fixed GitHub publication contract and four source
files; source hashes must match that contract. It separately compares two
model files and six kernel files, including two generated staging metadata
files. The staging digest and serialization follow the existing publisher's
algorithm without importing or executing it. CRLF and LF are not normalized.

Only anonymous HTTPS GETs to `raw.githubusercontent.com` and `huggingface.co`
are permitted. Each URL is constructed from a fixed repository, allowed path
and explicit immutable revision. Redirects must preserve the same typed
repository, commit and file. Up to three redirects are accepted; cycles,
credentials in URLs, alternate origins, mutable revisions, compressed bodies
and files over 256 KiB are rejected. Ambient HTTP proxies are not used.

This restricted transport intentionally does not follow arbitrary CDN/Xet
links or use credentials to read private files. An unavailable response is
not proof of absence. A network that requires an ambient proxy may produce
`TRANSPORT_UNAVAILABLE`; do not weaken the verifier or infer missing bytes.

## Result semantics

- `BYTE_ALIGNMENT_VERIFIED_REVIEW_REQUIRED`: all eight selected provider files
  were fetched and matched the contract-bound source and generated metadata.
- `BYTE_DRIFT_OBSERVED`: all comparisons completed and at least one differed.
- `OBSERVATION_INCOMPLETE`: at least one comparison was unavailable; any known
  matches and differences are retained.
- `OBSERVATION_REJECTED`: invalid input, invalid source contract or source-hash
  disagreement prevented completion.
- `NOT_OBSERVED`: no explicit network opt-in was supplied.

All outcomes keep `publication_verified`, `write_authorization_verified`,
`signature_independently_verified`, `branch_membership_verified` and
`freshness_verified` false. `repository_mutation` is `NOT_ATTEMPTED`.
Byte equality at supplied commits does not establish current branch membership,
provider write authorization, authenticated build provenance or model fitness.

Reports use atomic replacement and preserve partial observations. A failed
local write is not success. Keep generated reports outside the source checkout
and retain the exact command, pins, observation time and authenticated release
record. Do not use this CLI's exit code alone as promotion permission.

## Tests and release dependency

```powershell
python -m unittest discover -s tests -p test_invariants_hub_bytes.py -v
python -O -m unittest discover -s tests -p test_invariants_hub_bytes.py -v
```

The four-job offline workflow runs both modes on Windows/Linux and Python
3.11/3.12. Fixtures block network connections and are not live receipts.
Inspect actual exact-head results, full repository checks and source review
before normal protected merge. Do not apply the obsolete seven-file publisher
handoff over newer main.

The active publisher observed at this admission base still uses
`secrets.HF_ORG_TOKEN`; its credential and any separately reviewed keyless
integration remain tracked by #284. A refreshed chat connection or merely
setting an unused OIDC variable does not repair that workflow. After valid
CI authentication is established, perform one fresh canonical publication,
retain the real two-report artifact, and independently compare the resulting
immutable provider bytes. Do not mark #284 closed from this verifier's merge.
