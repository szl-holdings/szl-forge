---
title: OAC System Health Lab
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: apache-2.0
suggested_hardware: cpu-basic
models:
  - SZLHOLDINGS/oac-system-health-v1
  - SZLHOLDINGS/oac-ops-health-v2
datasets:
  - SZLHOLDINGS/oac-clinical-transport-observability-synthetic
short_description: Synthetic v1 and opt-in v2 telemetry advisories.
---

<p><a href="https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab"><img src="https://raw.githubusercontent.com/szl-holdings/.github/main/profile/assets/szl/logos/szl_mark_holographic.svg" alt="SZL Holdings" width="112" /></a></p>

# OAC System Health Lab

Explore fixed synthetic operational telemetry models through bounded stateless inputs.

**Artifact:** Synthetic telemetry demonstration · **Stage:** Research preview

[Explore in Command Lab](https://huggingface.co/spaces/SZLHOLDINGS/szl-command-lab) · [Build](https://github.com/szl-holdings/szl-forge) · [Evidence](https://github.com/szl-holdings/szl-forge/blob/bed8adf0e91f1506b81c8c27def6c9ca5aa7655e/spaces/oac-system-health-lab/README.md)

## Before you use it

- Do not submit clinical or sensitive data; this is not a PHI detector or a clinical decision system.
- No device connection, training, input persistence, receipt signing or medical authority is supplied.

## Opt-in v2 research preview

The existing v1 UI and `POST /api/score` remain the default. The separate v2
control must be explicitly enabled before this page requests
`GET /api/v2/identity` or `POST /api/v2/score`. v2 accepts the same eight bounded
synthetic operational fields inside exactly `{"features": {...}}` and returns an
`ALERT`, `NO_ALERT`, or `ABSTAIN` **advisory only**. All five authority flags are
false. No response acknowledges a transport message, controls a device,
interprets or releases a result, or makes a care decision.

The v2 Space package is pinned to the public
[`SZLHOLDINGS/oac-ops-health-v2` revision `a824a32d91a383d33a1e1e595f11b8362d1b4efa`](https://huggingface.co/SZLHOLDINGS/oac-ops-health-v2/tree/a824a32d91a383d33a1e1e595f11b8362d1b4efa)
and exact Forge artifact bytes at `56a00821858825f529c40c7322c2f1584608d6e5`.
`release_v2.json` binds the kernel, JSON coefficients, receipt, and authored
example by SHA-256. The kernel is a deterministic standard-library numeric
scorer, not a transformer checkpoint. Its evaluation is **REPORTED for
generated synthetic data only**; it has not been independently qualified for
real transports or clinical workflows. The source receipt's origin commit is
from an unpublished local repository; the public release verifies landed
bytes, not that private origin. `GET /api/v2/readyz` is v2-specific; v1
`GET /readyz` remains unchanged. A valid source binding and a successful score
are not production promotion or clinical validation.

<details>
<summary>Technical details and original evidence</summary>

The retained source below is exact and may contain historical observations. Its dates, use restrictions, licenses and evidence boundaries continue to apply.

<!-- SZL-PRESERVED-TECHNICAL-BODY:START -->

# OAC System Health Lab

A bounded, stateless, Python-standard-library **synthetic operational telemetry
preview**. It runs the existing OAC System Health v1 logistic model; it creates
no new model weights, performs no training, connects to no device, writes no
receipts or input files, and grants no clinical authority.

Canonical application source is
[szl-holdings/szl-forge](https://github.com/szl-holdings/szl-forge/tree/main/spaces/oac-system-health-lab).
The fixed model and kernel come from that repository's
`clinical-gateway/huggingface/model/oac-system-health-v1` package. The release
manifest identifies its immutable artifact source and the pinned
[model](https://huggingface.co/SZLHOLDINGS/oac-system-health-v1) and
[synthetic dataset](https://huggingface.co/datasets/SZLHOLDINGS/oac-clinical-transport-observability-synthetic)
revisions. The app's source revision is separately supplied by the governed
publisher in the nonsecret `SZL_GITHUB_SOURCE_REVISION` variable.

The server hashes all three fixed artifacts against independently pinned
digests **before importing the kernel**. An artifact drift, malformed release,
or missing/malformed application-source binding prevents readiness and scoring.
These hashes establish consistency with the declared release, not independent
authorship, reproducible training, model accuracy, a signature, or certification.

## Boundary

Use invented numerical telemetry only. Do not submit patients, PHI, personal
information, credentials, specimen identifiers, orders, results, HL7, FHIR,
assays, or actual site/device data. The closed request has exactly eight fixed
numeric/boolean feature fields. Unknown fields and nonnumeric values are
refused, not silently discarded. Numerical inputs alone cannot establish
whether a user has encoded sensitive information; this is not a PHI detector.

Scores are synthetic attention advisories, not production-calibrated
probabilities, clinical decisions, medical triage, laboratory validation,
result acceptance/release, transport acknowledgments, or device-control
instructions. The existing kernel returns explicit all-false authority flags.
No availability or score output changes those boundaries.

The app source deliberately does not log request paths, headers, bodies, or
input keys and does not intentionally persist submissions. Hosting/network
logging outside this source is not controlled or claimed absent. Runtime
outputs are unsigned; no authoritative receipt is minted. The public preview
has no authentication, tenant isolation, durability service, or SLA. Its
free-hourly `cpu-basic` target does not mean local electricity or hosting
resources have zero cost. The publisher does not request paid hardware.

Access Lab and the approved-document adapter retain their separate local-first
privacy and operating boundaries; this Space does not upload their state,
documents, repair requests, receipts, or credentials.

## API

- `GET /healthz`: process liveness, HTTP 200 even when the fixed model is
  unavailable. The JSON `state` must be inspected; this is not readiness.
- `GET /readyz`: HTTP 200 only with verified artifacts and a valid exact
  application source revision; otherwise HTTP 503.
- `GET /api/build-info`: observed application source binding or explicit
  `UNKNOWN`, runtime state, and `receipt_minted: false`.
- `GET /api/v1/identity`: application source, immutable artifact source,
  pinned Hub revisions and artifact digests, and all-false promotion/clinical
  declarations. Verified artifact metadata can remain visible when only the
  application-source binding is absent.
- `POST /api/score`: accepts `Content-Type: application/json`, one exact
  `Content-Length`, and the opt-in `X-SZL-Preview: 1` header. No chunked
  requests, CORS grants, redirects, file reads, URLs, free text, tools, or
  automatic external submission are supported.

Example invented request:

```json
{
  "features": {
    "listener_running": 1,
    "tls_enabled": 1,
    "peer_allowlist_configured": 1,
    "queue_utilization": 0.1,
    "consecutive_failures": 0,
    "seconds_since_last_success": 1,
    "ledger_integrity_ok": 1,
    "configuration_valid": 1
  }
}
```

Successful output has `ok`, `advisory`, `identity`, `receipt_minted: false`,
`input_sha256`, and `output_sha256`. The input hash covers the **features object
only**, not the wrapper or raw wire bytes. The output hash covers the **advisory
object only**, not identity metadata or the entire response. Both use UTF-8 JSON
with sorted keys, compact separators, unescaped Unicode, and finite numbers:
`json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False).encode("utf-8")`. Alternative whitespace/key order on the
wire therefore does not change a parsed-input hash; representations such as
integer `1` versus float `1.0` can differ. These hashes are ordinary unsigned
integrity evidence, not a replay proof or attestation.

## Runtime limits and security

One absolute **five-second receive deadline** begins when a connection is
accepted and covers both headers and body. Each underlying socket receive
uses the remaining budget, so a peer cannot renew the deadline by trickling
bytes. At most four connection threads run; accepted excess work receives
HTTP 503 without an unbounded application queue. All responses close their
connection. POST bodies are capped at 4,096 bytes, entire wire requests at
16,384 bytes, and JSON value nesting at eight levels. Duplicate keys, nonfinite
numbers, overlong integers, invalid UTF-8, and lone surrogates are rejected.
Refusal responses never echo input keys, values, exception messages, or paths.
Malformed/slow headers may be closed before an HTTP response can be formed;
body timeouts receive HTTP 408 when the connection remains writable.

The UI uses no external scripts, fonts, trackers, or model calls. The response
Content Security Policy hashes every inline script/style block and allows
network calls only to the same origin. Embedding is limited to this origin,
`https://a-11-oy.com`, and `https://a11oy.net`. This permits a separately reviewed
domain integration; it does not claim either domain is already deployed or
functionally aligned. The CSP and closed API are defense-in-depth, not an OS
sandbox, adversarial-host certification, or a production server SLA.

## Frontend design and response boundary

Prism Ledger uses opaque midnight content surfaces, static iridescent card
edges, and a decorative eight-spoke motif. The motif is not live telemetry.
Controls have explicit labels and visible keyboard focus; small actions use
44-pixel comfort targets. Narrow layouts stack numerical fields and long
identities wrap rather than clip. Reduced-motion and forced-color preferences
are respected. Static contrast/layout-token tests are not browser acceptance
or accessibility certification.

The browser refuses extra envelope, advisory, or identity fields and missing
or malformed SHA-256 strings. Hash syntax is not cryptographic verification;
the independent live verifier checks canonical input/output hash equality.
All prior authority, source-pin, request-bound, no-storage, and refusal
contracts remain unchanged.

## Local execution and tests

Python 3.11–3.14, no third-party runtime packages. From this directory:

```text
python -I -B app.py --host 127.0.0.1 --port 7860
python -I -B -m unittest discover -s tests -p "test_*.py" -v
```

Without a publisher-supplied exact source binding, liveness/UI remain
inspectable, but readiness and scoring are unavailable. Do not invent a source
revision to pretend a local checkout is a governed deployment. The Docker
entry point binds the public Space port; local examples explicitly bind
loopback. The tests use temporary trusted directories, bundled fixed
artifacts, invented inputs, loopback peers, and shortened receive budgets;
they do not connect medical devices or validate a clinical workflow.

Apache-2.0. The kernel, model and synthetic data retain their canonical license
and provenance. No independent medical, safety, regulatory, performance,
accessibility, or production certification is claimed.

<!-- SZL-PRESERVED-TECHNICAL-BODY:END -->

</details>
