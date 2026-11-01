# Public portfolio audit and bounded Nano research

These commands add public metadata coverage and synthetic CPU experiments. They
do not authorize model training against estate corpora, promotion, publication,
production inference, or GPU jobs. Existing admission and security controls remain
authoritative.

## Inventory and readback

`tools/audit_training_portfolio.py` enumerates models, datasets and Spaces for
each requested owner, follows pagination, rejects duplicate identities and
off-scope continuations, pins revisions, reads only README files, and repeats the
inventory. It never downloads model weights or dataset rows and never receives
tokens. A missing or changed repository cannot receive a stable-readback badge.
The explicit output directory must not already exist.

```sh
python tools/audit_training_portfolio.py --live \
  --owner SZLHOLDINGS --owner betterwithage --owner alloyscape \
  --output /tmp/szl-public-portfolio
```

The result is bounded to public API visibility at the recorded times. Private,
gated-content, local, deleted, and undiscovered repositories are not certified.
A gated repository's public card can be visible while its weight access is not.
Inventory lanes are metadata triage, not architecture attestation.

Badges deliberately have narrow meanings:

| Label | Evidence required | Does not establish |
|---|---|---|
| PUBLIC_REVISION_REOBSERVED | Complete first and second public listings; same identities and revision | Model quality or independent review |
| CARD_BYTES_READ_BACK | Matching immutable revision header and captured card SHA-256 | Factual accuracy of card claims |
| MEASURED_SYNTHETIC_LOCAL_ONLY | Named local experiment, source pins, splits, seed, counts and artifacts | Production quality, historical replay or promotion |

No independent-review badge is emitted. Both HTTP observations use the same
executor. Hashes provide byte integrity, not signatures or permission.

## Separate training, development and test

`tools/run_nano_research.py` uses exactly two Apache-2.0 source modules from
`szl-holdings/szl-khipu@8ba6893099aecfe1e7cced1b50075ccdd169e744`.
Obtain these exact text files into a separate directory and verify their hashes;
the runner refuses modified source before importing it:

| File in source directory | SHA-256 |
|---|---|
| receipt_agent.py | bda481e78238e001100356dbb40ef2a8dd53b3490942f2a810d6d0aac5408609 |
| tiny_khipu.py | 4a1929574477ea356897b8eb1aec6ca655fe03b964061eea34332c5dadfb5cbe |

The pinned paths in that repository are `szl_khipu/train/<filename>`.
The runner requires NumPy, uses seeds 11/23/37, caps ReceiptAgent at 600 steps
per seed, and TinyKhipu at 280 steps per seed. No production weights are loaded
or replaced.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_nano_research.py \
  --synthetic-cpu --source-directory /tmp/pinned-nano-source \
  --output /tmp/szl-nano-research
```

ReceiptAgent generates distinct 600/200/400-row training/development/test sets.
Exact row duplication and cross-split overlap are rejected. Development alone
selects checkpoints; the untouched test is evaluated after selection and
compared with the untrained initialization. Counts, confusion matrices, Wilson
intervals, advisory ALLOW errors, source hashes, tensor hashes and archive
readback are retained. This tests generalization within one synthetic generator
family, not across real data sources. The deterministic rule checker retains
decision authority.

TinyKhipu retains the pinned existing algorithm and adds a diagnostic: identical
query, matching versus unrelated handles, required opposite decisions. Its
decision head only reads query embeddings, so this exposes a structural
limitation that its existing plan-format metrics do not measure. This command
does not silently change the historical model architecture or publication.

Observed local experiment on 2026-11-01:

| Seed | ReceiptAgent correct/test N | Wilson 95% | Advisory ALLOW errors |
|---|---|---|---|
| 11 | 366/400 (91.5%) | 88.36–93.85% | 11 |
| 23 | 354/400 (88.5%) | 85.00–91.27% | 32 |
| 37 | 376/400 (94.0%) | 91.23–95.94% | 1 |

TinyKhipu's original synthetic metrics were plan_valid=1, abstain=1 and
hallucinated=0 for all three seeds; each counterfactual pair was only 1/2
correct. Neither model is promoted. A second process reproduced the tensor
digests and evaluation metrics. This is repeatability by the same executor,
not an independent evaluator.

## Applying published open training methods

- [Tulu 3, arXiv 2411.15124v5](https://arxiv.org/abs/2411.15124v5):
  separate development/unseen evaluations and decontamination; SFT, DPO and
  verifiable-reward RL are distinct stages. Start with measured SFT baselines;
  do not add RL before reward validity is independently tested.
- [Olmo 3, arXiv 2512.13961v2](https://arxiv.org/abs/2512.13961v2):
  retain stage/checkpoint/data/dependency lineage. Apply this to adapters,
  merges and quantized derivatives separately.
- [SmolLM3 engineering recipe](https://huggingface.co/blog/smollm3) and
  [open implementation](https://github.com/huggingface/smollm):
  use explicit stage-specific mixtures and evaluate improvements by domain.
  SZL should compare source-group ablations before combining corpora.
- [QLoRA, arXiv 2305.14314](https://arxiv.org/abs/2305.14314):
  frozen quantized base plus trainable adapters is a candidate for the small
  Qwen lanes. Published 48GB results do not prove current occupied A6000 capacity
  or compatibility with a different architecture.
- [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer):
  validate assistant/completion masking against the actual pinned chat template.
  Template requirements and library versions must be tested together.

NVIDIA's current [Nemotron 3 Nano recipe](https://docs.nvidia.com/nemotron/nightly/nemotron/nano3/README.html)
describes a 31.6B total / 3.6B active hybrid model. It must not be treated as an
architecture match for a locally reported 4B model solely because both names
contain Nemotron. Exact config, tokenizer, weights and dependency pins are
required. SZL's deterministic Nemo witness is a separate software component.

## Corpus use

The same-revision `SZLHOLDINGS/model-bom` training blocklist explicitly excludes
`killinchu-osint-corpus` and `szl-quant-sft-v1`. Its license register is historical
and does not cover all current datasets. A permissive repository license is not
a source-level rights audit. Benchmark, receipt and CI-evidence repositories
must not be poured into training because a Hub viewer calls their split
`train`. Source/document/derivation-family splits must be established before
chunking or synthesizing examples. Evaluation records remain evaluation evidence.

`publishing/training-research-map-20261101.json` records all observed model and
dataset identities and conservative research roles. It is an audit queue, not
an admission document, scheduler input, or replacement for existing controls.

## Validation

```sh
python -m unittest discover -s tests -p 'test_audit*.py' -v
python -m unittest discover -s tests -p test_nano_research.py -v
```

The first command includes the existing 60 evidence-auditor tests plus 15 public
inventory regressions. The second includes six research-boundary regressions.
The existing base Python CI collects these files; no live audit or training
command is added to CI. Full repository CI is a separate exact-head result.

