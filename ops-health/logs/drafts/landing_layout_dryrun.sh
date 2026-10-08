#!/usr/bin/env bash
# drafts: dry run of the card's three "Reproduce from source" commands in a staged copy of the
# proposed szl-forge ops-health/ layout. Local only: a throwaway git repo in the session
# scratchpad; nothing is pushed. SYNTHETIC.
# The staged tree is the committed bytes (git archive HEAD) of v2/, runs/ and math/, plus the
# untracked hub/oac-v1/ snapshot of v1's published files. v2/data/sealed/ is deliberately NOT
# copied: this run reads no sealed file, so make_data --verify is expected to fail on it.
# Usage: bash logs/drafts/landing_layout_dryrun.sh <stage_dir>
set -u
SRC=/c/Users/steph/szl-work/oac-frontier
STAGE="$1"
PY="py -3.12"
export PYTHONUTF8=1
echo "# drafts: landing-layout dry run; started $(date -u +%Y-%m-%dT%H:%M:%SZ); source HEAD $(git -C "$SRC" rev-parse HEAD)"
rm -rf "$STAGE"
mkdir -p "$STAGE/ops-health/hub"
git -C "$SRC" archive --format=tar HEAD v2 runs math | tar -x -C "$STAGE/ops-health"
rm -rf "$STAGE/ops-health/v2/data/sealed"
mkdir -p "$STAGE/ops-health/hub/oac-v1"
for f in LICENSE README.md artifact_receipt.json example_input.json model.json oac_operational_health.py; do
  cp "$SRC/hub/oac-v1/$f" "$STAGE/ops-health/hub/oac-v1/$f"
done
printf 'ops-health/** -text\n' > "$STAGE/.gitattributes"
printf 'scratch/\n__pycache__/\n' > "$STAGE/ops-health/.gitignore"
git -C "$STAGE" init -q
git -C "$STAGE" -c core.autocrlf=false add -A .
git -C "$STAGE" -c core.autocrlf=false -c user.name=dryrun -c user.email=dryrun@localhost commit -q -m "dry run"
echo "staged: $(git -C "$STAGE" ls-files | wc -l) files; sealed dir present: $(test -d "$STAGE/ops-health/v2/data/sealed" && echo yes || echo no)"
echo "du: $(du -sh "$STAGE/ops-health" | cut -f1)"
cd "$STAGE/ops-health" || exit 3
for f in v2/ops-health/ops_health.py v2/ops-health/model.json v2/ops-health/artifact_receipt.json v2/ops-health/example_input.json; do
  echo "sha256 $(sha256sum "$f")"
done

echo
echo '## 1/3 $ PYTHONUTF8=1 python -B -m v2.research.make_data --verify   (expected to fail: sealed/ not staged)'
$PY -B -m v2.research.make_data --verify 2>&1 | tail -4
echo "exit=${PIPESTATUS[0]}"

echo
echo '## 2/3 $ PYTHONUTF8=1 python -B -m unittest discover -s v2/tests -t .'
$PY -B -m unittest discover -s v2/tests -t . 2>&1 | tail -25
echo "exit=${PIPESTATUS[0]}"

echo
echo '## 3/3 $ PYTHONUTF8=1 python -B -m v2.research.final_analyze'
$PY -B -m v2.research.final_analyze 2>&1 | tail -15
echo "exit=${PIPESTATUS[0]}"

echo
echo '## compare the rewritten final_evaluation.json with the committed one (numeric leaves)'
git show HEAD:ops-health/v2/results/final_evaluation.json > "$STAGE/committed_eval.json"
$PY -B - "$STAGE/committed_eval.json" v2/results/final_evaluation.json <<'EOF'
import json, sys
def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{p}/{i}")
    else:
        yield p, o
a = dict(leaves(json.load(open(sys.argv[1], encoding="utf-8"))))
b = dict(leaves(json.load(open(sys.argv[2], encoding="utf-8"))))
num = [k for k, v in a.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
timing = [k for k in num if k.endswith("/seconds")]
cmp = [k for k in num if k not in timing]
same = sum(1 for k in cmp if b.get(k) == a[k])
other = sorted(k for k, v in a.items() if k not in num and b.get(k) != v)
print(f"leaves committed {len(a)} rerun {len(b)}; numeric compared {len(cmp)}, identical {same}; "
      f"timing fields excluded {len(timing)}")
print(f"non-numeric leaves that differ: {other}")
print("verdict committed", a.get("/verdict/result"), "rerun", b.get("/verdict/result"))
EOF
echo "git status after the run (files the analysis rewrote):"
git -C "$STAGE" status --short
echo "# finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
