# ReceiptAgent v3 public-card correction input

The preparer is a review input, not a model publication or release. The public
`SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3` Hub card at immutable revision
`f4b28d75e1bdbbbf299bb8f5beceb8f141f38baf` correctly retains the historical
11/12 DEV check, but wrongly says the Forge card's 12/12 row contradicted it.
The Forge authoring card at `receiptagent-v3/card/README.md` distinguishes the
2026-08-29 11/12 check from the 2026-09-29 additive adapter-bound 12/12 check.
The latter has `base_model=UNRECORDED`, `promotion_effect=NONE`, and no held-out
or merged-checkpoint qualification. Both remain owner-run DEV n=12 records.
The old Hub YAML summarized only the historical check, and its bottom source
note falsely said the current Forge card still asserted a single 12/12 record.

`tools/prepare_receiptagent_v3_card_sync.py` pins the reviewed Forge card blob,
the observed Hub parent and README digest, then checks both DEV receipts at
that parent. It changes exactly five reviewed spans in `README.md`: the YAML
evaluation summary, the development-check table, its explanation, the evidence
row, and the bottom source note. The candidate SHA-256 is
`79fa68b40bcd8cdfff3c4c0a50a7f4c452bf62b080fa95080baffb02a9430555`
(10,885 bytes). Every other Hub README byte is preserved. Missing, duplicated, partially changed or
already edited spans fail closed. Hub parent drift also fails closed. No weight
file is fetched or loaded.

From a Forge checkout containing the exact source commit, run:

```sh
python -B -m unittest discover -s tests -p test_receiptagent_v3_card_sync.py -v
python -B tools/prepare_receiptagent_v3_card_sync.py --source-revision <40-hex-git-commit>
# Optional local review artifacts; the output directory must not already exist.
python -B tools/prepare_receiptagent_v3_card_sync.py --source-revision <40-hex-git-commit> --output-dir card-review
```

The default command writes nothing. The optional directory contains only a
candidate `README.md`, `review.diff` and `receipt.json`. Its receipt says
`REVIEW_ONLY_NO_HUB_WRITE` and `UNQUALIFIED`. The preparer has no Hub credential,
commit, upload, training, inference, rights clearance or promotion path. The
source commit in that receipt is an exact byte binding, not an assertion that
the checkout is current protected `main` or that publication was authorized.

The existing `tools/publish_receiptagent_v3.py` targets the separate
`szl-receiptagent-qwen35-0.8b-v3-authenticated` repository; it does not own this
public card. The curated-card route currently admits WILLAY only.

`tools/publish_receiptagent_v3_public_card.py` is a distinct, one-shot writer
for this exact public target. Its default command is review-only. The explicit
`--publish` path requires a new exclusive local receipt path, a supplied exact
source revision that matches both checkout HEAD and freshly queried protected
Forge main, and committed publisher/preparer bytes. It reruns the five-anchor
preparer, checks the pinned public Hub parent and old README digest, authenticates
write scope, and uses exactly one `README.md` addition with `parent_commit`.
The reviewed `huggingface_hub==1.23.0` commit-origin marker is required because
the SDK may otherwise return another writer's head without sending a CAS. No
ambiguous commit is retried. Immutable README readback and a final Hub-head
check are required before the receipt can say `PUBLISHED_AND_READ_BACK`.
Any post-attempt uncertainty says `POST_COMMIT_UNVERIFIED_DO_NOT_RETRY` and
requires manual reconciliation. The receipt never grants model qualification.

The fresh-main observation is an identity guard, not proof of branch protection,
required CI, signed merge, or owner approval. Those remain separate release
checks. After the source PR is reviewed and merged, an authorized operator can run the
review-only command from the exact current main. The publish form is documented
here for the controlled release window, **not** as evidence it has run:

```sh
python -B tools/publish_receiptagent_v3_public_card.py --source-revision <current-main-sha>
# Only after owner diff review, an HF write grant, and the reviewed 1.23.0 SDK:
python -B tools/publish_receiptagent_v3_public_card.py --source-revision <current-main-sha> --publish --receipt-file <new-local-receipt.json>
```

The development shell observed on 2026-10-02 has SDK 1.33.0 and no `HF_TOKEN`;
it is **not** an authorized publication environment. A changed Hub parent
requires a new reviewed candidate and digest, not a forced write. Existing
license, lineage, consent, privacy, training-suitability, deployment and model
promotion holds remain untouched.

## Manual GitHub publication workflow

After this source is merged, `publish-receiptagent-v3-public-card.yml` offers
one explicit dispatch on current protected `main`. It has no push or PR writer.
Supply the exact current main SHA, the reviewed Hub parent above, and the literal
`README_ONLY_UNQUALIFIED` confirmation. The workflow validates all three before
credentials, runs the offline tests and preparer, then checks current main again.
The publisher, preparer, dispatch guard, credential selector and workflow must
all match their committed source. The credential selector actively verifies
write permission for this exact model; configured secret names alone are not
evidence of access. Only the reviewed Hub client version is installed.

```sh
gh workflow run publish-receiptagent-v3-public-card.yml \
  --repo szl-holdings/szl-forge --ref main \
  -f source_revision=<current-main-40-hex-sha> \
  -f expected_hub_parent=f4b28d75e1bdbbbf299bb8f5beceb8f141f38baf \
  -f confirmation=README_ONLY_UNQUALIFIED
```

The per-model concurrency lock does not cancel an in-progress writer. Evidence
is uploaded even when a control fails. GitHub's Re-run action is rejected because
`GITHUB_RUN_ATTEMPT` must equal `1`. Do not start a fresh dispatch to repeat an
ambiguous attempt: first reconcile the prior receipt and immutable Hub bytes.
A fresh dispatch is not a globally unique consumed intent. If main moves after
a Hub commit, the receipt can remain `POST_COMMIT_UNVERIFIED_DO_NOT_RETRY` even
though the card was written; a failed workflow does not prove no write occurred.

The successful state for this workflow is a corrected README with matching
immutable bytes. Every model qualification and deployment hold still applies.
