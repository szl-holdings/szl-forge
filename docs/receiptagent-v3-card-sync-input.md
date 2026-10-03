# ReceiptAgent v3 public-card correction input

This is a review input, not a model publication or release. The public
`SZLHOLDINGS/szl-receiptagent-qwen35-0.8b-v3` Hub card at immutable revision
`f4b28d75e1bdbbbf299bb8f5beceb8f141f38baf` correctly retains the historical
11/12 DEV check, but wrongly says the Forge card's 12/12 row contradicted it.
The Forge authoring card at `receiptagent-v3/card/README.md` distinguishes the
2026-08-29 11/12 check from the 2026-09-29 additive adapter-bound 12/12 check.
The latter has `base_model=UNRECORDED`, `promotion_effect=NONE`, and no held-out
or merged-checkpoint qualification. Both remain owner-run DEV n=12 records.

`tools/prepare_receiptagent_v3_card_sync.py` pins the reviewed Forge card blob,
the observed Hub parent and README digest, then checks both DEV receipts at
that parent. It changes only three exact stale prose spans in `README.md` and
preserves all other Hub README bytes. Missing, duplicated, partially changed or
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
public card. The curated-card route currently admits WILLAY only. Before this
candidate can be published, the owner must review the full managed/unmanaged
diff and explicitly establish a publisher for this exact target. That route
must requalify fresh protected source, require the exact current Hub parent,
use provider compare-and-swap, change only `README.md`, read every changed byte
back at the immutable resulting revision, and retain partial-write failure
evidence. A changed Hub parent requires a new reviewed candidate. The existing
license, lineage, consent, privacy, training-suitability, deployment and model
promotion holds remain untouched.
