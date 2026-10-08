"""drafts: post-hoc cross-split duplicate check for REFORMS item 6b of drafts/model_card_v2.md.

Not registered in v2/PREREGISTRATION.md or v2/AMENDMENTS.md; descriptive only; SYNTHETIC.
Reads only the unsealed development splits (v2/data/{train,calibration,conformal,validation}.jsonl)
and the saved post-opening per-row score files (v2/results/final/scores_*.jsonl), which record each
scored row's eight features. It never opens anything under v2/data/sealed/ and imports no v2 code,
so no sealed guard is involved. Read-only: it writes nothing; stdout is the log.

Run: PYTHONUTF8=1 py -3.12 -B logs/drafts/dup_check_from_scores.py
"""
import hashlib
import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("listener_running", "tls_enabled", "peer_allowlist_configured", "queue_utilization",
          "consecutive_failures", "seconds_since_last_success", "ledger_integrity_ok",
          "configuration_valid")
DEV = ("train", "calibration", "conformal", "validation")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vectors(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [tuple(json.loads(line)["features"][k] for k in FIELDS) for line in fh if line.strip()]


print("# drafts: post-hoc cross-split duplicate check (exact 8-feature vectors). "
      "MEASURED, post hoc, NOT registered; SYNTHETIC. No file under v2/data/sealed/ is read.")
sets = {}
for name in DEV:
    p = ROOT / "v2" / "data" / f"{name}.jsonl"
    sets[name] = vectors(p)
    print(f"input {p.relative_to(ROOT).as_posix()} sha256 {sha256(p)}")
scored = []
for p in sorted((ROOT / "v2" / "results" / "final").glob("scores_*.jsonl")):
    sets[p.stem] = vectors(p)
    scored.append(p.stem)
    print(f"input {p.relative_to(ROOT).as_posix()} sha256 {sha256(p)}")

for name, ks in sets.items():
    print(f"{name}: rows {len(ks)} unique feature vectors {len(set(ks))}")
dev_union = set().union(*(set(sets[n]) for n in DEV))
dev_rows = sum(len(sets[n]) for n in DEV)
scored_rows = sum(len(sets[n]) for n in scored)
total_hits = 0
for name in scored:
    hits = sum(1 for k in sets[name] if k in dev_union)
    total_hits += hits
    print(f"{name}: exact matches against train/calibration/conformal/validation: {hits}")
for a, b in combinations(DEV, 2):
    print(f"dev pair {a} x {b}: shared vectors {len(set(sets[a]) & set(sets[b]))}")
for a, b in combinations(scored, 2):
    print(f"scored pair {a} x {b}: shared vectors {len(set(sets[a]) & set(sets[b]))}")
print(f"SUMMARY scored rows {scored_rows} in {len(scored)} sets; development rows {dev_rows}; "
      f"exact matches scored-vs-development {total_hits}")
