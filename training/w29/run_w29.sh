#!/usr/bin/env bash
# Wave 29: train only the allowlisted causal-LM candidate repos.
# Protected live non-causal-LM repos are never passed to train_wave.py.
# A qlora tag on any other target refuses the wave before training starts.
# W29_PLAN=1 prints the same decisions and does not train or push.
# W29_TRAINER overrides the trainer executable (tests). Protected repos never reach it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="${W29_CONFIG_DIR:-$ROOT}"
TRAINER="${W29_TRAINER:-$ROOT/../train_wave.py}"
cd "$CONFIG_DIR"

if ! python3 "$ROOT/safety.py" preflight "$CONFIG_DIR"; then
  exit 1
fi

shopt -s nullglob
for cfg in "$CONFIG_DIR"/*.json; do
  name=$(basename "$cfg" .json)
  line=$(python3 "$ROOT/safety.py" decide "$cfg")
  action="${line%%$'\t'*}"
  reason="${line#*$'\t'}"
  if [ "$action" = "SKIP" ]; then
    echo "SKIP $name ($reason)"
    continue
  fi
  if [ "$action" != "RUN" ]; then
    echo "REFUSE $name ($reason)" >&2
    exit 1
  fi
  if [ "${W29_PLAN:-0}" = "1" ]; then
    echo "PLAN $name ($reason)"
    continue
  fi
  echo "=== w29: $name ==="
  model=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["base_model"])' "$cfg")
  datasets=$(python3 -c 'import json,sys; print(",".join(json.load(open(sys.argv[1], encoding="utf-8"))["datasets"]))' "$cfg")
  epochs=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["epochs"])' "$cfg")
  seqlen=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["seqlen"])' "$cfg")
  repo=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["target_repo"])' "$cfg")
  python3 "$TRAINER" \
    --model "$model" \
    --datasets "$datasets" \
    --method qlora \
    --epochs "$epochs" \
    --seqlen "$seqlen" \
    --pack \
    --repo-id "$repo" \
    --log-repo-id SZLHOLDINGS/wave-logs \
    --out "./out/$name"
done
