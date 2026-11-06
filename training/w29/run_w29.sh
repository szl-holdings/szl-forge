#!/usr/bin/env bash
# Wave 29: train the untrained estate model candidates.
# Run on an A6000-class node with HF_TOKEN set. One model at a time.
set -euo pipefail
cd "$(dirname "$0")"
for cfg in *.json; do
  name=$(basename "$cfg" .json)
  method=$(python3 -c "import json;print(json.load(open('$cfg'))['method'])")
  if [ "$method" != "qlora" ]; then
    echo "SKIP $name (method=$method; requires a specialized trainer — see README)"
    continue
  fi
  echo "=== w29: $name ==="
  python3 ../train_wave.py \
    --model "$(python3 -c "import json;print(json.load(open('$cfg'))['base_model'])")" \
    --datasets "$(python3 -c "import json;print(','.join(json.load(open('$cfg'))['datasets']))")" \
    --method qlora --epochs "$(python3 -c "import json;print(json.load(open('$cfg'))['epochs'])")" \
    --seqlen "$(python3 -c "import json;print(json.load(open('$cfg'))['seqlen'])")" --pack \
    --repo-id "$(python3 -c "import json;print(json.load(open('$cfg'))['target_repo'])")" \
    --log-repo-id SZLHOLDINGS/wave-logs \
    --out "./out/$name"
done
