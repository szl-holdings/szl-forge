# OAC evidence-first model audit demo

This is a public, synthetic research demonstration to discuss with scientists.
It is not a Roche cobas Liat device interface, a patient-result path, a
laboratory validation, a clinical deployment, or an Anthropic endorsement.

The [public SZL frontier catalog](https://holdings.a-11-oy.com/frontier/) is a
dated index of GitHub and Hugging Face assets. A model-type repository may be a
checkpoint, adapter, GGUF derivative, software kernel, numeric fixture,
training recipe, or placeholder. Catalog presence and README claims do not
qualify model weights. The catalog's own receipt states its observation time;
this demo does not silently update it or claim every repository was executed.

The specific runnable case is the [OAC System Health Lab](https://szlholdings-oac-system-health-lab.hf.space/),
also linked from the [A11oy OAC page](https://a-11-oy.com/oac) and the
[proof page](https://a11oy.net/oac/). It scores invented eight-field numerical
telemetry only. The logistic-regression coefficient JSON is pinned to
[Hub revision `dd7d1098…`](https://huggingface.co/SZLHOLDINGS/oac-system-health-v1/tree/dd7d109813abcd90c5250106dc2eabd2804e7ac3);
its dataset is [synthetic revision `f7ab6170…`](https://huggingface.co/datasets/SZLHOLDINGS/oac-clinical-transport-observability-synthetic/tree/f7ab6170bf78138b187b8cb707d374a25ad85375).
The separately reviewed six model payload files matched immutable
[Forge source `234a2dfb…`](https://github.com/szl-holdings/szl-forge/tree/234a2dfb4c511318febfd20ca2f695fa9cb2ea8c/clinical-gateway/huggingface/model/oac-system-health-v1).
That source/Hub audit is separate from this runner's three-artifact Git-source
parity and live-runtime checks; no provider byte attestation is inferred from
the runner alone.

From a Forge checkout with Python 3.11 or newer and network access:

```text
python -I -B demos/oac-audit-demo/run.py
```

The runner imports this checkout's canonical OAC verifier. That verifier checks
three packaged artifacts against immutable Git source, live readiness and
declared source/model/dataset identity, two invented scores against the local
kernel, canonical input/output hashes, and refusal of a clinical-like extra
field without echo. This wrapper refuses a changed source pin, missing probe,
failed authority boundary, or altered artifact digest, and prints only a
closed public-safe summary. It writes no files, downloads no weights, asks for
no credential, and does not create a clinical receipt. Failure exits nonzero.

For the presentation, show one score in the Lab UI, then the runner's seven
checks. Also show a real limitation: the published 240-row **synthetic** test
reports balanced accuracy 0.7414 and ROC AUC 0.8302, while the documented
queue-saturation shift collapses fixed-threshold balanced accuracy to 0.5.
Those figures are not independently replayed by this demo and are not
production or clinical performance estimates.

## Research question

Can an evidence-aware assistant distinguish a loadable model from a kernel,
fixture, script-only repository, empty placeholder, or release-blocked
experiment, while keeping missing evidence `UNKNOWN`? A next-stage study would
pin every public model revision, verify loader/base/adapter-key coverage where
appropriate, replay named held-out gates, and compare its error rate against a
metadata-only catalog. Private assets require separate authorization and must
not enter a public projection. This one OAC witness is a working exemplar,
not a claim that the entire estate passed.

Anthropic says its [AI for Science program](https://www.anthropic.com/news/ai-for-science-program)
supports high-impact research and its [scientist tools](https://www.anthropic.com/news/expanding-support-for-scientists)
emphasize auditable artifacts. A submission or collaboration is the owner's
decision and must satisfy the [official rules](https://www.anthropic.com/ai-for-science-program-rules),
including rights and any institutional consents. This repository claims no
Anthropic or Stanford affiliation, award, credits, or endorsement.

For actual Roche clinical connectivity, a separate private laboratory-approved
project needs exact analyzer/software and assay versions, authorized interface
documents, LIS contracts, patient/order/specimen binding, QC/exception and
critical-result workflows, privacy/security review, and witnessed site
validation. None is established by a public synthetic demo.
