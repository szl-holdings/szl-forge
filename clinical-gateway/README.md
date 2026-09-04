# Owned Agent Clinical Shadow Stack

This stack exposes a local control API, a configuration-only browser shell, and
bounded result-ingress transports around the Owned Agent clinical kernel.

It is **not a clinical system**. LIVE_SHADOW means deidentified live transport
observation only. It does not mean a validated device connection, patient
identity resolution, autoverification, result release, LIS/EHR delivery,
clinical-use authorization, HIPAA compliance, or site acceptance.

## Truth boundary

| Surface | What it proves | What it does not prove |
| --- | --- | --- |
| MOCK | The fixed synthetic fixture path can run | Any physical analyzer, real result, or clinical workflow |
| LIVE_SHADOW | A configured transport can submit deidentified input to the local review kernel | Identity, clinical correctness, device authenticity, site validation, or delivery |
| GET /api/health | The local API process answered | Analyzer connectivity, storage integrity, or clinical readiness |
| Transport running=true | The local listener/worker thread is running | That a Liat connected, TLS trust is correct, or messages were accepted |
| Offline FHIR candidate | A deterministic local transformation was produced | Endpoint conformance, EHR acceptance, chart insertion, or result release |
| Local audit/hash records | Local tamper evidence under the implemented boundary | External immutability, independent approval, or a regulated audit service |
| Operational-health advisory | A verified local artifact scored eight bounded infrastructure signals | Production calibration, device state, patient risk, diagnostic validity, ACK/release authority, or clinical readiness |

Always preserve clinical_use_authorized=false, real_phi_authorized=false, and
site_validated=false.

## Components

- src/owned_agent_clinical_control.py: bounded MOCK and deidentified
  LIVE_SHADOW review/export kernel. It does not issue device commands or deliver
  results.
- src/oac_live_transport_bridge.py: separate live-shadow transport runtime.
- src/oac_stack_integration.py: programmatic orchestration and local
  operational evidence.
- src/oac_stack_api.py: loopback-only HTTP control API with required bearer
  authentication and origin allowlisting.
- src/oac_operational_health.py: verified, advisory-only logistic-regression
  inference kernel for synthetic operational transport signals.
- frontend/index.html: control/configuration UI. It has no
  raw-message, patient, order, specimen, observation, or result input.
- fixtures/assay_map.json: exact MOCK mapping accepted by the core.
- fixtures/assay_map.live-shadow.example.json: non-production site mapping
  placeholder.
- fixtures/live-shadow-bindings/: deidentified binding-directory example.
- fixtures/live_shadow_site.example.json: copy-safe configuration shape
  with loopback, example paths, and no secrets or PHI.
- docs/CLEAN_ROOM_STANDARDS.md: source, attribution, and independent
  implementation rules.
- operational-model/: fixed-seed synthetic splits, closed schema, model,
  receipts, example, and reproduction instructions.
- huggingface/: exact closed-manifest model and dataset upload staging trees.

## Model, kernel, and dataset boundary

The control engine uses `EnrichedContextGenerator`, `EntropyDepthAllocator`,
`CrossStepConsistency`, `LoopKernelStateTransition`, and `ClinicalPolicyCheck`
only as deterministic validation and evidence machinery. None of them assigns
clinical confidence, interprets a result, overrides a failed gate, or decides
whether a result should be released.

`ControlEvidenceModel` is a deterministic operational score over command
outcomes. It cannot feed back into intake, review, authorization, FHIR content,
or transport acknowledgements. Each observation is appended to a local JSONL
file under schema `owned-agent-clinical-control/operational-observation/v1` and
is labeled `deterministic_operational_evidence_not_clinical_confidence`. This is
an allowlisted local observability log, not patient data, clinical evidence, a
training corpus, or proof of model performance. Do not fine-tune a model on it
and do not put PHI into it.

The separately trained `OperationalHealthKernel` accepts exactly eight bounded
transport/configuration features and returns an operator-attention advisory. It
uses 1,200 fixed-seed synthetic examples and standard-library logistic
regression. Its held-out synthetic test ROC AUC is 0.830169104679 and its
precision is 0.412371134021 at the validation-selected threshold; those numbers
are deterministic implementation checks only, not production or clinical
performance. The output contains an explicit all-false authority map and is not
called by ingestion, MLLP ACK handling, clinical review, FHIR generation, or
release logic.

## Documented Roche transport profile

The roche-cobas-liat-v2.0 preset is a local adapter name. It is configured
against the official **cobas liat system Host Interface Manual HL7,
Publication v11.3, software 3.4 and 3.5, May 2025**.

The supported direction and message flow are:

    cobas Liat TCP client -> this host TCP listener
    MLLP: VT + HL7 payload + FS + CR
    ORU^R30^ORU_R30 -> ACK^R33^ACK
    MSH-12 literal: 2.5

The direct interface documented by Roche is result reporting. This integration
does not implement host orders, analyzer control, assay execution, patient
notification, or undocumented commands.

ACK handling is bounded as follows:

- AA is sent only after the local ingest callback reports successful
  processing.
- Malformed framing, an invalid envelope, an oversized frame, the wrong message
  type, or the wrong configured HL7 version receives AR.
- Internal ingest rejection, queue/backpressure, or a stopping transport
  receives AE because the host could not process/store the accepted frame.
- MSA-2 echoes the inbound MSH-10. AE and AR include an ERR segment with a local
  error code.

Roche documents up to three send attempts. Lost-ACK behavior, retransmission,
duplicate control IDs, exact ACK replay, and recovery after a process failure
must be tested with the physical analyzer. The current implementation is not
site-validated for those scenarios.

The public US manual covers software 3.4/3.5. Roche advertises software 4.0 in
some CE markets, but no public official 4.0 host-interface manual was located
during this build. Do not infer 4.0 compatibility.

## TLS and network boundary

For the documented topology, the Liat is the TLS client and this stack is the
TLS server. The analyzer/operator trusts the host server certificate. The
preset therefore requires:

- tls_enabled=true;
- an existing server certificate and private-key file;
- minimum TLS 1.2;
- one or more literal analyzer IPs in allowed_peer_ips;
- one message per connection; and
- an explicit binding_dir.

The public Roche manual does not establish analyzer client-certificate support.
Leave tls_require_client_cert=false unless the exact analyzer/software/site
combination has been validated for mutual TLS. An IP allowlist and network
location are not cryptographic device identity.

For a physical analyzer, use a dedicated clinical integration VLAN, host
firewall rules, a non-public listener address, controlled certificate
rotation, synchronized clocks, monitoring, and downtime reconciliation. Never
expose the MLLP listener to the public Internet.

The file-drop adapter ignores symlinks and resolved paths outside its watch
directory. Read-only HTTP polling rejects URL credentials and all redirects so
a bearer credential cannot cross origins. Non-loopback plaintext MLLP or HTTP
requires an explicit insecure-transport acknowledgement and is not enabled by
the Roche TLS preset.

## Start the local control plane

Create an isolated environment and install the exact source tree:

    py -3.12 -m venv .venv-oac-clinical
    .\.venv-oac-clinical\Scripts\python.exe -m pip install --requirement .\clinical-gateway\requirements.lock
    .\.venv-oac-clinical\Scripts\python.exe -m pip install --no-deps .\clinical-gateway
    .\.venv-oac-clinical\Scripts\owned-agent-clinical-control.exe --version
    .\.venv-oac-clinical\Scripts\owned-agent-clinical-control.exe clinical-capabilities
    .\.venv-oac-clinical\Scripts\oac-operational-health.exe --model clinical-gateway\operational-model\artifacts\model.json --receipt clinical-gateway\operational-model\artifacts\model-receipt.json --input clinical-gateway\operational-model\example-input.json

From the repository root in PowerShell:

    $env:OAC_API_KEY = "<strong random value from your secret manager>"
    $env:OAC_ALLOWED_ORIGINS = "http://127.0.0.1:8010,http://127.0.0.1:8080"
    .\.venv-oac-clinical\Scripts\oac-clinical-gateway.exe --host 127.0.0.1 --port 8010 --data-root . --state-dir .runtime\oac-clinical-state --ui-file clinical-gateway\frontend\index.html

OAC_API_KEY is mandatory and must contain at least 32 characters. Obtain it
from an approved secret manager; do not commit it or put it in a fixture. The
API itself is loopback-only; use an authenticated TLS reverse proxy if a
separately approved deployment needs another access pattern.

The preferred same-origin shell is then available at
http://127.0.0.1:8010/. The backend URL field remains
http://127.0.0.1:8010.

When `--data-root` is the repository root, the server automatically loads the
hash-verified operational model and receipt from
`clinical-gateway/operational-model/artifacts/`. Otherwise provide both
`--operational-model` and `--operational-receipt`; providing only one fails
closed. Authenticated clients can inspect `/api/operational-health` and submit
the exact feature object to `/api/operational-health/score`. The browser shell
exposes typed operational fields only.

For isolated frontend development only, serve the file separately:

    python -m http.server 8080 --directory clinical-gateway\frontend --bind 127.0.0.1

Open http://127.0.0.1:8080/index.html only when using that
separate development server. Enter the bearer token and check
health/capabilities. The page does not persist the token in localStorage or
cookies. Do not use the page for PHI.

## MOCK workflow

1. Select MOCK.
2. Initialize state.
3. Register the fixed synthetic source.
4. Use only the core's synthetic fixtures and command-line test workflow.

The exact accepted MOCK assay-map shape is:

    {
      "SYNTH-FLU": {
        "display": "Synthetic influenza assay",
        "local_system": "urn:synthetic:assay"
      }
    }

The core intentionally rejects alternate MOCK mappings.

## LIVE_SHADOW workflow

LIVE_SHADOW must contain no direct identifiers and is not authorized for
clinical use.

1. Confirm the instrument region, exact software version, assay scripts, and
   matching Roche documentation.
2. Replace the example source's sender_application and sender_facility with the
   exact expected MSH-3 and MSH-4 values.
3. Build a laboratory-approved assay map. Its JSON must be a nonempty object
   keyed by the exact OBR-4 source code. Every value must contain exactly
   display and local_system; the system must be an HTTP(S) or URN URI.
4. Deidentify before ingress. The browser must never receive the HL7 message or
   its clinical content.
5. Provision the server-side binding registry. Each file must have the exact
   binding shape accepted by the core and a filename computed as SHA-256 over
   UTF-8 PID-3 + NUL + OBR-2. OBR-3 is used only when OBR-2 is empty. The hash
   is only a lookup index.
6. Install the server certificate/key, set the literal analyzer peer IPs, and
   start the Roche listener.
7. Inspect transport status as operational telemetry only. Confirm all required
   cases on an isolated test network before any broader deployment.

The supplied LIVE_SHADOW binding fixture uses only pseudonymous DEID values and
synthetic=false, deidentified=true. It is an executable shape example, not a
real-site mapping.

## Control API

| Method and route | Purpose | Authentication |
| --- | --- | --- |
| GET /api/health | Local process liveness and truth flags | Health exception; origin still checked |
| GET /api/capabilities | Kernel/API capability report | Bearer token |
| POST /api/command | Allowlisted clinical-kernel commands | Bearer token |
| GET /api/transports | List in-process transports | Bearer token |
| GET /api/transport/status | One/all transport status | Bearer token |
| POST /api/transport/start | Start a bounded transport | Bearer token |
| POST /api/transport/stop | Stop a bounded transport | Bearer token |

The browser intentionally does not expose dataset tails, arbitrary command
JSON, raw ingestion, review signatures, or FHIR contents.

## Clinical-result representation boundary

The intended offline FHIR R4 shape is one DiagnosticReport for a run/panel,
with atomic Observation resources for targets and a Device reference for
verified analyzer provenance. A separately emitted Ct OBX should remain a
separate numeric Observation. Patient, ServiceRequest, and Specimen references
must come from an authoritative, site-validated identity/order workflow.

Do not silently convert Invalid, Indeterminate, or Aborted to a negative/normal
result. Preserve the source value and route it to exception review. Roche's
LOINC document provides candidates, but the laboratory remains responsible for
the exact specimen/assay mapping. No FHIR delivery endpoint is implemented by
this stack.

## Required validation before any clinical claim

- Exact analyzer model, country/region, software, assay, and host-interface
  manual are pinned.
- Laboratory leadership approves assay/target terminology, units, specimen
  types, reference ranges, and exception handling.
- Patient/order/specimen matching is authoritative; PID-3 ambiguity is resolved.
- Positive, negative, invalid, indeterminate, aborted, corrected, and partial
  results are exercised.
- MLLP fragmentation, oversize/malformed frames, duplicate IDs, lost ACKs,
  reconnects, timeouts, backpressure, and failover are tested.
- Server-certificate issuance, trust, rotation, expiry, clock skew, cipher
  policy, and network isolation are tested with the physical analyzer.
- Downtime, reconciliation, correction, retention, authorized-recipient, audit,
  and critical-result procedures are approved and witnessed.
- FHIR conformance is validated against the target endpoint's current
  CapabilityStatement and local implementation guide.
- Security threat modeling, dependency inventory, SBOM, vulnerability handling,
  backup/restore, incident response, and access reviews are operational.
- Regulatory counsel determines intended use, including whether the software
  remains a transfer/display function or becomes device software.

Until those conditions are witnessed, the correct status is:
**DEIDENTIFIED LIVE SHADOW — NOT CLINICAL USE — NOT SITE VALIDATED**.

## Official primary sources

- Roche Diagnostics, [cobas liat system Host Interface Manual HL7, Publication
  v11.3, SW 3.4/3.5, May
  2025](https://diagnostics.roche.com/content/dam/diagnostics/us/en/products/c/cobas-liat-support/host-interface-manual-hl7-sw-v3.4-v3.5-v11.3.pdf)
- Roche Diagnostics, [US cobas Liat self-service support
  index](https://diagnostics.roche.com/us/en/product-sub-pages/c0/cobas-liat-self-service.html)
- Roche Diagnostics, [cobas liat System LOINC Code Listing, TP-01489 V2, June
  2025](https://diagnostics.roche.com/content/dam/diagnostics/us/en/products/c/cobas-liat-support/TP-01489-V2.pdf)
- IHE, [Pathology and Laboratory Medicine Technical Framework Revision
  11.0](https://profiles.ihe.net/PaLM/index.html) and [Volume
  2c](https://www.ihe.net/uploadedFiles/Documents/PaLM/IHE_PaLM_TF_Vol2c.pdf)
- HL7, [FHIR R4 DiagnosticReport](https://hl7.org/fhir/R4/diagnosticreport.html),
  [Observation](https://hl7.org/fhir/R4/observation.html),
  [Device](https://hl7.org/fhir/R4/device-definitions.html), and [transaction
  interaction](https://hl7.org/fhir/R4/http.html#transaction)
- HL7, [US Core Laboratory DiagnosticReport
  profile](https://hl7.org/fhir/us/core/StructureDefinition-us-core-diagnosticreport-lab.html)
  and [Laboratory Observation
  profile](https://www.hl7.org/fhir/us/core/StructureDefinition-us-core-observation-lab.html)
- CMS, [State Operations Manual Appendix C, Revision
  233](https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/downloads/som107ap_c_lab.pdf)
- FDA, [Cybersecurity in Medical Devices, final guidance, February
  2026](https://www.fda.gov/media/119933/download)
- FDA, [Design Considerations for Interoperable Medical
  Devices](https://www.fda.gov/media/95636/download?attachment=)
- FDA, [Medical Device Data
  Systems](https://www.fda.gov/medical-devices/general-hospital-devices-and-supplies/medical-device-data-systems)
- HHS, [HIPAA Security Rule
  summary](https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html)
- NIST, [SP 800-66 Revision 2](https://csrc.nist.gov/pubs/sp/800/66/r2/final),
  [SP 800-207 Zero Trust Architecture](https://csrc.nist.gov/pubs/sp/800/207/final),
  and [SP 800-218 SSDF 1.1](https://csrc.nist.gov/pubs/sp/800/218/final)

These sources support engineering and validation controls; they do not by
themselves establish legal compliance, regulatory classification, or clinical
authorization.

## Development verification

From the repository root with the runtime requirements installed:

    $env:PYTHONPATH = (Resolve-Path .\clinical-gateway\src).Path
    python -B -m unittest discover -s .\clinical-gateway\tests -v
    python -B -m compileall -q .\clinical-gateway\src .\clinical-gateway\tests
    python -m build --wheel .\clinical-gateway

CI runs the portable contracts on Python 3.11, 3.12, and 3.13, builds and
installs the wheel, and re-runs the contracts on Windows Server 2022. These are
source and isolated-runtime checks; they do not establish device or clinical
validation.
