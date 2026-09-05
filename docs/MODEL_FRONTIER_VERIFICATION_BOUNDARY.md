# Model frontier verification boundaries

## Failure repaired

The `Compile and run offline contracts` step invoked a live source-binding verifier. That verifier downloads exact-revision signed JSON receipts from the Hub, so it could not run anonymously once ReceiptAgent became gated. A missing public download is not permission to remove the gate or to call unverified receipts valid.

## Source verification

Pull requests run the existing Linux source, signature, authorization, portfolio, runtime-contract, kernel-container, and Windows release tests without Hub secrets. The local portfolio report is explicitly `OFFLINE_SOURCE`; it does not establish remote byte parity. Space packaging dry runs remain source-only. A green pull-request source job is not a live model release approval.

## Authenticated Hub evidence

The `Authenticated Hub receipts and exact source bindings` job runs only for this repository's main-branch push or main-branch manual invocation, after both source-verification jobs succeed. It checks out the exact event SHA, not an untrusted pull-request branch. Credentials are introduced only after verification-client installation and offline credential-selector tests.

The job reuses the already configured owner credential family and existing actively validated selector. The selector checks the existing ReceiptAgent repository; it does not create a repository, change gating, request access, mint a replacement secret, or publish model files. The existing selector verifies owner/write access, but this job's actual operations are read-only and never pass `--publish`.

Both prior remote checks still run and remain fail-closed: exact-source binding and signed-receipt verification, then live portfolio parity. Missing credentials, denied gated access, changed artifacts, invalid signatures, broken receipt chains, or mismatched source bytes fail the job. No exception is converted into a passing receipt.

The separate publisher is unchanged. This workflow grants no model execution, autonomous action, external targeting, training, allocation change, or publication authority. Public repository inventory and authenticated artifact verification are different evidence types.

## Evidence

Source reports: `model-portfolio-offline.json`, `model-inference-lab-dry-run.json`, and `szl-forge-lab-dry-run.json`.

Authenticated reports: `hf-model-verifier-credential.json` (secret-free selection evidence), `model-source-bindings-dry-run.json`, and `model-portfolio-live.json`. The legacy portfolio report mode `LIVE_PUBLIC_HUB` identifies the public portfolio being inspected; the containing authenticated job identifies the credential scope. Protected receipt bodies are not published by the job.

A successful authenticated run proves only the checks implemented by the source verifiers. Signed training claims are not independent benchmarks, and a signed aggregate weight digest is not a local recomputation of model weights. Runtime readiness and real inference require their own existing governed verification.

The nine `tools/test_model_frontier_auth_boundary.py` regressions check the event guard, exact source checkout, prerequisite gates, secret isolation, preserved cryptographic tests, retained remote checks, immutable actions, and absence of publication or model execution.
