# Clean-Room Interoperability Standard

## Purpose

This project implements documented interoperability behavior independently. It
may learn architectural patterns from public, vendor-owned documentation, but
it must not copy proprietary source code, private protocols, user-interface
expression, test data, confidential material, or substantial manual text.

Roche, cobas, Liat, HL7, FHIR, IHE, and other names are used only to identify
interoperability targets and standards. Their use does not imply ownership,
endorsement, partnership, certification, or site validation.

## Permitted evidence

Implementation decisions may be based on:

- official manufacturer interface and operator documentation for the exact
  device, software version, region, and assay;
- official standards publications and implementation guides;
- official regulator and government guidance;
- independently created synthetic or deidentified test fixtures; and
- behavior witnessed on equipment the operator is authorized to test.

Do not infer undocumented commands, fields, security mechanisms, or software
version compatibility. Mark missing behavior UNKNOWN and obtain the applicable
manufacturer/site document before implementation.

For every adapter, record:

- document title, publisher, publication/version/date, and direct URL;
- supported claim and the exact implementation decision it enables;
- exact device/software/assay/site scope;
- known contradictions, regional differences, and unresolved gaps; and
- validation evidence distinct from source, unit tests, CI, deployment, and
  witnessed runtime.

## Clean-room workflow

1. One research lane extracts only necessary externally documented behavior
   from primary sources and records attribution.
2. An implementation lane works from those behavior records and public
   standards, not copied vendor code or private materials.
3. Fixtures use synthetic or approved deidentified values. They must not reuse
   real patient, accession, sample, operator, device-secret, certificate, or
   credential data.
4. Review verifies that names, comments, UI, docs, and tests do not reproduce
   proprietary expression or make unsupported compatibility claims.
5. Physical-device and site validation is recorded separately. Passing source
   tests never upgrades a result to clinical-ready.

## Current cobas Liat boundary

The only implemented direct-device profile is documented result ingress for
public Roche software 3.4/3.5 host-interface manual v11.3:

- analyzer acts as TCP/TLS client and the host acts as listener;
- MLLP framing is VT, payload, FS, CR;
- inbound message is ORU^R30^ORU_R30;
- MSH-12 is the literal 2.5;
- response is ACK^R33^ACK;
- AA, AE, and AR retain their documented acceptance/error/rejection roles; and
- device commands and host-order workflows are not implemented.

Official source:
[Roche cobas liat Host Interface Manual HL7 v11.3, May
2025](https://diagnostics.roche.com/content/dam/diagnostics/us/en/products/c/cobas-liat-support/host-interface-manual-hl7-sw-v3.4-v3.5-v11.3.pdf).

No public official software 4.0 host-interface manual was found during this
build. Software 4.0 compatibility remains UNKNOWN. The public manual describes
the analyzer trusting the host/server certificate; it does not establish
analyzer client-certificate support. Mutual TLS remains site/version dependent.

## Clinical and regulatory separation

Transport, normalization, identity matching, terminology mapping, clinical
review, release, and delivery are separate stages. The implementation must fail
closed between them.

Models, agent kernels, entropy allocation, ranking, or autonomous reasoning
must not change, accept, suppress, interpret, release, or prioritize patient
results unless intended use, regulatory classification, clinical validation,
human governance, and postmarket controls have been established.

Invalid, Indeterminate, and Aborted are source states, not negative results.
PID-3 is not sufficient identity proof because the Roche documentation labels
it Patient / Sample ID. A site-authoritative patient/order/specimen match is a
hard publication gate.

Relevant official sources:

- [HL7 FHIR R4 DiagnosticReport](https://hl7.org/fhir/R4/diagnosticreport.html)
  and [Observation](https://hl7.org/fhir/R4/observation.html)
- [CMS State Operations Manual Appendix C, Revision
  233](https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/downloads/som107ap_c_lab.pdf)
- [FDA Cybersecurity in Medical Devices, February
  2026](https://www.fda.gov/media/119933/download)
- [FDA Interoperable Medical Devices
  guidance](https://www.fda.gov/media/95636/download?attachment=)
- [FDA Medical Device Data
  Systems](https://www.fda.gov/medical-devices/general-hospital-devices-and-supplies/medical-device-data-systems)

This documentation is engineering guidance, not a legal determination.

## Architecture-pattern research

The following vendor-owned pages were reviewed only for high-level
architectural patterns. This is a representative set, not a verified market
ranking:

- Roche, [navify point-of-care digital
  solutions](https://diagnostics.roche.com/us/en/products/product-category/lab-type/point-of-care-testing-poct/digital-solutions.html):
  separation of transport from diagnosis and support for managed heterogeneous
  device estates.
- Roche, [POC gateway system
  requirements](https://diagnostics.roche.com/content/dam/diagnostics/us/en/products/c/cobas-liat-support/poc-module-for-navify-integrator-system-requirements.pdf):
  local gateway with a centrally managed backend and durable local components.
- Data Innovations, [Instrument
  Manager](https://topics.datainnovations.com/product/instrument-manager/):
  versioned driver boundaries, centralized connections, staged rules, QC,
  permissions, and availability/recovery controls.
- Siemens Healthineers, [Atellica Data
  Manager](https://www.siemens-healthineers.com/en-us/diagnostics-it/productivity/atellica-data-manager):
  multiple LIS/site connections, laboratory-defined QC holds and flags, and
  review by exception. Some scale/driver statistics on the page use older
  footnotes and are not adopted as current claims.
- Abbott, [AlinIQ
  AMS](https://www.corelaboratory.abbott/us/en/offerings/brands/aliniq/aliniq-ams.html):
  analyzer/LIS connectivity, controlled rules, QC history, multisite workflow,
  monitoring, and driver architecture.

The independently selected pattern for this project is:

    versioned device adapter
      -> immutable bounded ingress receipt
      -> durable queue and idempotency control
      -> identity, QC, terminology, and policy gates
      -> human exception review
      -> explicitly authorized publication/delivery adapter

The current project stops before clinical publication/delivery.

## Prohibited representations

Do not claim:

- clinical use, FDA clearance/approval, CLIA validation, HIPAA compliance, or
  certification without applicable external evidence;
- live device operation from a listening socket, HTTP 200, dashboard badge, or
  CI result;
- device authentication from an IP allowlist;
- patient identity from a hash, PID-3, or pseudonym alone;
- FHIR conformance from producing JSON;
- vendor endorsement or ownership of vendor technology; or
- originality for copied or closely imitated proprietary expression.

Use accurate language: independently implemented from cited public
manufacturer and standards documentation, with unresolved items labeled
UNKNOWN, UNAVAILABLE, INCOMPLETE, BLOCKED, or FAILED_CLOSED as applicable.
