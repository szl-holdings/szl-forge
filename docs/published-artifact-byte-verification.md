# Verify published ReceiptAgent v2 bytes

The immutable `SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2` adapter can be checked as opaque bytes. This gate reads a local snapshot from the exact Hub revision in [model-source-bindings.json](../publishing/model-source-bindings.json). It never imports the model, executes remote code, runs inference, trains, or publishes.

The existing Forge receipt format is Ed25519 over canonical JSON. The tool calls Forge's maintained receipt-chain verifier, then asks the installed OpenSSL `pkeyutl -verify -rawin` to check the same three signed canonical byte strings. It checks the public key against the committed Forge source, the training/evaluation/publication hash links, six immutable source/Hub file pairs, adapter metadata, and an actual streamed SHA-256 over the adapter file. This is a separate format from the DSSE/P-256 benchmark in `szl-receipt`.

## Replay

Get the required eight small files and the adapter from the **exact** declared Hub commit. The local snapshot must contain `hub-metadata.json` from the Hub model-information response, plus `candidate.json`, `owner_pubkey.json`, `publication.json`, `adapter_config.json`, `adapter_model.safetensors`, and the three files under `receipts/`. Use an isolated directory outside the Git worktree. Do not replace a signed SKU or run a model loader.

On Windows with Git for Windows OpenSSL:

```powershell
py -3 -B -X utf8 tools/verify_published_artifact_bytes.py `
  --snapshot 'C:/path/to/immutable-v2-snapshot' `
  --repo-id 'SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v2' `
  --revision 'bd642c7ff18736248e84fd83dace7ab368fc2288' `
  --source-revision (git rev-parse HEAD) `
  --openssl 'C:/Program Files/Git/usr/bin/openssl.exe' `
  --output 'C:/path/to/verification-receipt.json'
```

The command prints a SHA-256 of the exact UTF-8 output bytes. Exit 0 means the specified byte and signature checks matched this repository's **declared** pins. Exit 1 is BLOCKED when data disagrees; exit 2 is UNAVAILABLE when a required file, source object, library, or OpenSSL verifier is absent. The output states `key_trust: REPO_DECLARED`, `publication_eligible: false`, and `production_authorization: BLOCKED` even on exit 0.

The snapshot provider origin is a caller declaration; the local command does not download from the Hub. A separate retrieval receipt must prove the exact provider revision and downloaded byte provenance. The CLI report measures one adapter and its declared receipt set, not every file in the Hub repository. The owner-signed historical qualification source, numerical equivalence, held-out evaluation replay, runtime readiness, independent signer identity, and model quality are outside this gate.

The public v2 adapter used for the initial readback was 43,346,432 bytes with SHA-256 `885fc29fcb4cf55c280dc085fdb0a40f40d6b946fee400dd5e4ed3459fe6334f` at Hub commit `bd642c7ff18736248e84fd83dace7ab368fc2288`. The repository-declared key fingerprint is an integrity comparison, not an out-of-band trust decision. The current model remains proposal-only with limited evidence.

Run the SAMPLE negative controls without downloading an artifact:

```powershell
py -3 -B -X utf8 -m unittest discover -s tests -p test_published_artifact_bytes.py -v
```

The tests use ephemeral SAMPLE signing keys and verify that tampered bytes, metadata-only inputs, an unadmitted valid signer, unsigned/noncanonical receipts, broken signed links, wrong revisions, modified source, and missing/rejecting OpenSSL fail closed.
