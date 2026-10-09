# Offline artifact metadata checks

Use Python 3.11 or later from the repository root. This mode uses only the standard
library; it does not require a Hub token, fetch repositories, load weights, verify
signed receipts, or execute models.

```sh
python -B tools/verify_model_portfolio.py --artifact-observation observation.json --report report.json
```

The input is a UTF-8 local JSON object with schema `szl.artifact-observation/v1`.
Required fields are `revision`, `inventory_complete`, and `files`; optional fields
are `json_observations` and `gguf_header_observations`. Additional envelope keys
are rejected. Each file record supplies `path`, `revision`, and `size`. Metadata
observations are keyed by exact repository-relative path and supply `revision`,
`state`, and decoded `data` when `state` is `PARSED`. The classifier's docstring
lists supported indexes, states, and metadata shapes.

This synthetic example describes metadata, not an observed or qualified model:

```json
{
  "schema": "szl.artifact-observation/v1",
  "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "inventory_complete": true,
  "files": [
    {"path": "config.json", "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 8},
    {"path": "model.safetensors", "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 8}
  ],
  "json_observations": {
    "config.json": {
      "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "state": "PARSED",
      "data": {"model_type": "synthetic"}
    }
  }
}
```

Every inventory record and metadata envelope must bind to the same lowercase
40-character commit. A conflicting config cannot establish adapter lineage; a
conflicting index, payload, or GGUF record cannot establish its package's complete
structure. Duplicate inventory paths also invalidate their local claims in either
record order. Supply `null` for unobserved revision, completeness, or size rather than
inventing a value. Only a complete, consistently bound inventory establishes absence.

The JSON report has schema `szl.artifact-observation-report/v1`. Its
`observation_sha256` hashes the entire input file's exact bytes, including
whitespace. It records reproducibility; `source_authenticity` remains
`NOT_EVALUATED`. Inputs larger than 8 MiB, duplicate keys, invalid encodings,
non-finite numbers, malformed JSON, and invalid envelopes fail closed. The report
path must differ from the input, including existing symlink and hard-link aliases.

Exit code `0` means `COMPLETE_STRUCTURE` for the supplied principal-weight package
metadata only. Unknown or incomplete metadata and invalid inputs return `1`;
incompatible command options return `2`. Detailed observation codes stay in
`metadata_structure`. `artifact_validity` and `runtime_validity` remain
`NOT_EVALUATED`; no tensor integrity, load compatibility, training, evaluation,
rights, promotion, deployment, or release approval follows from this result.

Run the focused synthetic regressions from `tools/`:

```sh
python -B -m unittest test_verify_model_portfolio.ArtifactCompletenessTests test_verify_model_portfolio.ArtifactObservationCliTests
```

The CLI fixtures block provider imports, network connections, and child process
creation inside the checker. The two existing live-audit unit fixtures in
`ArtifactCompletenessTests` use mocked Hub metadata and require the project's
existing test dependency `huggingface-hub`; the CLI fixtures require only Python.
