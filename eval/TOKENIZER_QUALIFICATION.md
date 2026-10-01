# Tokenizer compatibility and Python ingress measurements

## Invariant and owner

This is a local observation/comparison tool beside Forge's existing `eval/`
programs. Speed cannot compensate for a different token sequence, span mapping,
mask, overflow, or decoded result. No baseline, production dependency, route,
training job, publication gate, or release authority is changed. The existing
base Python gate discovers the offline tests in `tests/`; the narrowly scoped
native-control workflow covers real library calls on Windows/Linux 3.11/3.12.

Primary upstream sources, inspected 2026-09-22:
- https://huggingface.co/blog/tokenizers-v1
- https://huggingface.co/docs/tokenizers/api/tokenizer
- https://github.com/huggingface/tokenizers/blob/main/REQUIRED_FOR_V1.md
- https://pypi.org/project/tokenizers/0.23.2/#files

The v1 article measures Rust calls, not Python binding or application overhead.
Its stated ID compatibility does not discharge our rich-encoding contract.
Missing RC Python APIs are `UNAVAILABLE`, not two equal missing values. Do not
install an invented v1 Python version. Obtain and independently review an exact
available build, source, toolchain, wheel digest and dependency lock before use.

## Actual Python interface

`python eval/tokenizer_qualification.py identity` identifies the installed
Python package/native binary hashes. It is an observation, not package-source
attestation. Run it in each separately prepared environment. Never replace the
production environment to run this experiment.

`capture` reads only an explicit local tokenizer JSON and corpus. Both require
expected SHA-256 values. The JSON used by the engine is the exact in-memory byte
snapshot hashed by the observer, avoiding a second tokenizer-file read. Native
version and module hash must match supplied expectations. Capture requires a
fresh interpreter and sets the requested Rayon count before importing tokenizers.
No installer, HTTP client, model loader, subprocess or secret reader is present
in the observer. A trusted native library can itself execute code: use an
isolated no-secret/no-network sandbox, not this script as a security sandbox.

Corpus shape (all strings are synthetic in this example):

```json
{"schema":"szl.tokenizer-corpus/v1","cases":[
 {"id":"empty","text":"","pair":null},
 {"id":"code","text":"print(1)\n","pair":null},
 {"id":"unicode-pair","text":"café 雪 🛰️","pair":"paired passage"}
]}
```

Limits: 4 MiB corpus, 256 cases, 32 KiB UTF-8 per text/pair, 8 MiB tokenizer
JSON, 3–30 timed repeats, requested thread counts 1/2/4/8. Native outputs are
bounded after each call; a pathological native call is not preempted. Use the
outer job/OS deadline and memory limits. CI's 12-minute job and synthetic test's
60-second child deadline are outside the observer. Windows report privacy needs
appropriate directory ACLs in addition to exclusive file creation.

The capture covers both special-token settings, single and batch calls,
IDs, token strings, offsets, masks, type/word/sequence IDs, overflow trees and
both decode settings. The exact source JSON's padding/truncation/normalization
configuration is retained. Single and batch padding can legitimately differ;
we compare each call mode against its own oracle, never strip padding away.
A complete feature matrix requires captures for each intended configuration.
This does not test raw bytes as text, custom Python callbacks, training,
streaming APIs, chat-template rendering or special multimodal processor wrappers.
Pre-render chat templates through their pinned owner and separately validate
that wrapper; do not silently claim the tokenizer-only result covers it.

In each fresh environment, substitute real approved input/build digests:

```bash
python eval/tokenizer_qualification.py capture \
  --tokenizer /private/tokenizer.json --tokenizer-sha256 "$TOKENIZER_SHA256" \
  --corpus /private/corpus.json --corpus-sha256 "$CORPUS_SHA256" \
  --source "$REVIEWED_FORGE_REVISION" \
  --version "$EXPECTED_VERSION" --native-sha256 "$EXPECTED_NATIVE_SHA256" \
  --threads 1 --repeats 5 --output /private/new-baseline.json
```

Run an independent candidate process against the SAME source/input hashes;
repeat independently for 1/2/4/8 threads. Then compare the observations:

```bash
python eval/tokenizer_qualification.py compare \
  --baseline /private/new-baseline.json --baseline-sha256 "$BASELINE_FILE_SHA256" \
  --candidate /private/new-candidate.json --candidate-sha256 "$CANDIDATE_FILE_SHA256" \
  --output /private/new-comparison.json
```

No example SHA is a valid pin. `--source` is explicitly operator-declared;
`observer_sha256` identifies actual script bytes. Preserve the job's actual
commit, run and attempt externally via the existing receipt authority. Hashes
are integrity identifiers, not signatures, privacy clearance or anonymization.
No raw text/token IDs or local paths are written to observations. Case labels,
host hashes and timing patterns can still reveal information: keep reports
private until the existing closed public projection reviews them.

## Measurement meaning and limits

Load time is separate from Python encode calls. First-sweep timings reuse word
cache across documents; only the first encode starts after a fresh tokenizer
load. Subsequent samples are labeled `REPEATED_FIXED_CORPUS_PYTHON_CALLS`: they
are deliberately NOT a large distinct-document/cache-cold benchmark. Each warm
batch's full output is checked outside the timing loop. Decode calls are timed
separately and rechecked. Setup and hashing overhead are excluded from encode
but included in process RSS high-water. RSS is bytes on Linux/macOS; unavailable
elsewhere is null, not zero. Requested threads are not measured physical-core
placement or proof the engine used them.

Comparator rows are matched by case ID, special-token mode and call mode.
Unknown fields, missing repeats, duplicate rows, source/input mismatch and
nonfinite values reject. Any field mismatch emits `FAIL_PARITY` and withholds
performance ratios. Matching same-build observations are labeled a CONTROL,
not a candidate benefit. Ratios are descriptive medians only and require matching
observed environment fields and requested threads; these do not prove fixed
CPU frequency, quiet neighbors, physical cores, same host custody or comparable
thermal conditions. A digest-recomputed forged report is not authenticated
execution evidence. Native run/attempt/byte attestation remains external.

The observer never declares a general speedup or enables promotion. To establish
benefit, retain distinct-document/shared-prefix/unique-text workloads, randomized
AB/BA fresh-process runs, sufficient samples, host-to-host repetitions, physical
core/power controls, and the intended actual tokenizer/model families. Compute
throughput/percentiles from retained samples without timing validation work.
Measure end-to-end ingestion/backpressure, TTFT and GPU utilization separately.

## Integration order

1. Admit the observer source/tests through normal Forge protection.
2. Freeze owned/de-identified code, legal, maritime and observability corpus
   slices and exact tokenizer/normalizer/processor profiles. No customer data
   or benchmark holdouts become training material.
3. Capture current oracle and exact available RC in isolated environments.
   Unsupported profiles remain blocked; do not downgrade to ID-only comparison.
4. Qualify corpus preparation, then retrieval indexing, then batch prefill,
   then interactive traffic, each with its own reviewed evidence and rollback.
   Rebuild indexes only when their own tokenization/index contract changes.
5. Extend the existing product's evidence reader only after adopting this
   schema through its actual owner. Do not create another API/workbench or
   treat these private files as a ready-made public API.
6. Publish permitted artifacts through GitHub → Hugging Face → a-11-oy.com →
   a11oy.net. Tokenizer evaluation is not a model release or training permission.

The intended frontier improvement is measured ingress efficiency with preserved
semantic and provenance contracts, not an unsupported new performance claim.

## Tests and synthetic native control

```bash
python -m unittest discover -s tests -p test_tokenizer_qualification.py -v
python -O -m unittest discover -s tests -p test_tokenizer_qualification.py -v
```

The offline suite uses explicit deterministic doubles. It does not import a
native tokenizer. The dedicated workflow separately installs only the two
hash-pinned 0.23.2 test-wheel alternatives in disposable runners, then executes
actual single/batch/decode calls in fresh processes. Synthetic WordLevel fixtures
cover Unicode, pairs, specials, empty text, code, long text, truncation/overflow,
fixed padding and batch padding across 1/2/4/8 requested threads. A deliberately
mutated observation must fail comparison. These controls test the evaluator,
not v1, representative vocabularies, BPE/Unigram quality or production benefit.

The workflow preserves partial reports on failure and has read-only permissions,
no secrets, dispatch, schedules, model downloads, GPU requests or publisher.
Artifacts have 14-day retention, not immutable custody. Protect longer-term
release receipts via the established system.

Exit 0 means a capture was recorded or a declared paired profile matched;
exit 2 means unavailable native/profile execution or observed parity mismatch;
exit 1 means malformed/unreadable inputs or local output failure. Existing output
files are never overwritten. Abruptly terminated empty files are invalid evidence.
