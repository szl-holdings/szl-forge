# Salad candidate evaluation selection

## Invariants before implementation

The original dataset revision and byte hashes are immutable. Preserve all original training rows. An evaluation row is eligible only if none of its complete normalized user messages occurs anywhere in the original training data. Deduplicate eligible evaluation rows by the ordered complete normalized user-message context, retaining the first original row. Final assistant answers never participate in exclusion or deduplication decisions.

Record both the 1-based physical JSONL line and 0-based nonblank record index, together with the original raw-line SHA256 (including its original newline bytes). Reject unsupported message roles, ignored message metadata, nonalternating conversations, empty supervised targets, and assistant targets without preceding user input. Assert zero retained user-turn, full-prompt, and assistant-target-prefix overlap.

The tool must not change the source corpus, create a training job, publish a model, or declare a trained model qualified. Source hashes and counts must pass before any output is written. A separately requested materialization may write the selected original lines into a new local file, with a verified digest; it must never overwrite either input.

## Why this is needed

The original v4 manifest describes 26,706 training rows and 2,800 evaluation rows. Its file hashes and counts verify, but complete normalized prompt comparison found 529 shared contexts, affecting 687 evaluation rows. Whole-example comparison found no shared examples because the corresponding answers differ. Checking only whole examples would therefore miss the prompt exposure.

The stricter any-user-turn policy independently identified 691 affected evaluation rows. Removing those and seven remaining duplicate prompt identities leaves 2,102 rows. The original training set is preserved. This is an exact-match-filtered residual of the existing evaluation split; independent generalization is **NOT ESTABLISHED**.

## Reproduction

The source is the private Hugging Face dataset SZLHOLDINGS/szl-frontier-sft-mix-v4, revision 12a26b44303299624592084c025e344d260aa7bf. Download data/sft_train.jsonl and data/eval_repo.jsonl at that revision using the existing authenticated Hugging Face client. Do not copy credentials or data into this repository.

Run the standard-library-only selector with Python 3.12 using Unicode database 15.0.0:

    python tools/salad_eval_selection.py --spec docs/salad-eval-selection-20261116.inputs.json --train <private-cache>/sft_train.jsonl --eval <private-cache>/eval_repo.jsonl --output-dir docs/receipts/salad-eval-selection-20261116

The outputs contain identities, counts, hashes, and limitations; they contain no raw prompt or answer text. The optional --materialize-eval <new-private-path> writes the selected original evaluation lines outside the source files. No training or network operation is implemented by this tool.

## Interpretation

NFKC, case folding, and whitespace collapsing are conservative equality rules. They can merge distinctions that matter in code and mathematics. The selector records the normalizer ID and Unicode database version. Matching is never truncated to a prefix.

The selection does not establish semantic or near-duplicate independence, source-family separation, consent or license coverage, PII handling, or absence from pretraining. Removing exposed rows now cannot erase an existing adapter's historical exposure. The proposed next candidate must start from the pinned pristine base, retain the private output repository, and remain held until its evaluation and governance plan is reviewed. No existing job's dataset is silently replaced.

The two original Salad jobs have a separately verified dependency/authentication repair and default to preflight while stopped. Their one-row and 48-row corpora remain unchanged. The existing OAuth credential expires on 2026-12-14 at 00:00:50 UTC; restoring it is not durable credential provisioning.

## Observed message metadata

The pinned source contains role/content messages and messages with an additional exactly empty tool_calls list. Only that inert optional field is permitted. Active calls, null/non-list values, names, and other ignored metadata are rejected. The pinned Qwen tokenizer is checked to serialize omitted and empty tool_calls identically; raw line hashes still bind the original bytes.

## Residual distribution and validation

The retained STEM stratum has only 10 rows, down from 569 original evaluation rows (556 exposed rows and 3 residual duplicate rows removed). Aggregate residual loss cannot establish broad STEM capability. Report source strata separately and use a separately governed capability evaluation before promotion.

Thirteen targeted tests cover hash/count/Unicode drift, changes between initial hashing and parsing, late-turn overlap, multiple matching turns counted once per row, complete long prompts, changed answers, strict message metadata, raw newline identity, exclusive receipt creation, and selection/materialization binding. The pinned tokenizer serializes omission and exactly empty tool_calls identically at both text and token-ID levels. A one-step CPU Qwen3/LoRA compatibility smoke separately passed; no GPU qualification or production training is claimed.
