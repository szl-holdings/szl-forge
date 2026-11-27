# proofrun (vendored)

Proof-carrying training-run receipts and their verifier, vendored for
`.github/workflows/proofrun-receipt-attest.yml`.

- Source: Rikuy's proofrun package (SZL internal, upstream tree hash `8e10c1d7...c156`). Changes: an unused
  `write_json` import was removed from `synthetic.py` (ruff F401), and report details no longer echo
  `--trusted-key` / `--trusted-identity` values (CodeQL py/clear-text-logging-sensitive-data).
- Tree hash: `a65e2009eafa1101d3c93c9d34296ee09aecafc3c7f8b50659016a4c9cd0ab0d`
  (`find proofrun -name '*.py' | sort | xargs sha256sum | sha256sum`, run in this directory).
- `requirements-ci.txt`: hash-pinned wheels (cryptography and deps) for CPython 3.12 on ubuntu-24.04 x86_64,
  installed with `pip install --require-hashes`.

The workflow attests custody of a finished run bundle with Sigstore keyless signing: a GitHub OIDC token,
a short-lived Fulcio certificate, an entry in the public Rekor log, and an in-toto statement in a DSSE envelope.
The subject is `proofrun_bundle.json` and the predicate type is
`https://szl-holdings.github.io/proofrun/endorsement/v1`. The endorsement never upgrades a claim: anything
not passed by the AQ-16 gate stays NOT QUALIFIED.

Verify a run:

```
python3 -m proofrun verify <run_dir> \
  --trusted-identity https://github.com/szl-holdings/szl-forge/.github/workflows/proofrun-receipt-attest.yml@refs/heads/main
```
