# frontier-sft-mix-v2 (Git twin)

**Hub canonical:** [SZLHOLDINGS/szl-frontier-sft-mix-v2](https://huggingface.co/datasets/SZLHOLDINGS/szl-frontier-sft-mix-v2)

Credit: Stephen Lutar (Stephen P. Lutar Jr.) / SZL Holdings  
Doctrine v11 LOCKED · Λ = Conjecture 1 (advisory) · MEASURED-or-null  
`training_allowed=true` · `private_graph_admitted=false`

Thin twin of the Hugging Face dataset for forge-local validation and pin tracking.
Prefer Hub revision digests for training; this tree is a receipted mirror of the first cut.

## Validate

```bash
python tools/validate_sft_dataset.py data/frontier-sft-mix-v2/train.jsonl \
  --disjoint-from data/frontier-sft-mix-v2/val.jsonl --min-examples 8
```

## Pins

See `SHA256SUMS`, `recipe.json`, `manifest.json`, and `build_summary.json`.

No AlloyScape production merge. No silent GPU train without Rosa TRAIN PLAN + GO.
