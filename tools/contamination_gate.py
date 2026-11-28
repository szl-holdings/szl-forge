#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contamination gate for SFT/DPO training launches (proposal).

Compares the model *inputs* (system + user turns; never assistant labels) of a
train file against:
  * a sealed holdout, whose manifest pins path, sha256, row count and group ids;
  * every reference corpus listed (prior mixes, prior validation splits, benches).

Three match levels are reported separately:
  exact       sha256 of NFC/strip-normalized role+content (same as
              tools/validate_sft_dataset.py, so results agree with the existing gate)
  normalized  NFKC + casefold, zero-width removed, punctuation -> space,
              whitespace collapsed; computed on the full input AND on the last
              user turn alone (catches a shared prompt under a different system prompt)
  near        character 5-gram Jaccard >= threshold on the normalized last user
              turn. This is LEXICAL near-duplicate detection only. It does not
              detect paraphrase, translation or semantic leakage.

Transformed-copy checks (added after W30 triage found bench cases reformatted
into training rows that share no verbatim prompt with the bench):
  id_leak     any reference identifier (case_id / id / problem_id, >= 4 chars)
              appearing in ANY train message, including assistant turns
  ngram       reference rows whose distinctive normalized word 8-grams (system
              prompts and template grams in > 50% of rows ignored) are >= 50%
              contained in train message text
  source      train rows whose source_dataset is a declared evaluation dataset id

Group disjointness: if rows carry a group key (default: source_dataset +
source_revision, else source_row_sha256), train and holdout groups must not intersect.

Verdict is REJECT on any exact/normalized/near overlap with the holdout or a
reference, on any group overlap, on a holdout whose bytes differ from its
sealed manifest, or on any unreadable/missing input. A clean result is
CLEAN_NOT_LAUNCHED: it never authorizes a launch by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA = "szl.forge.contamination-gate/v1"
MANIFEST_SCHEMA = "szl.forge.sealed-holdout/v1"
INPUT_ROLES = ("system", "user")
TEXT_FIELDS = ("input", "prompt", "question", "instruction", "scenario", "problem", "task")
_ZW = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZW.sub("", text).casefold()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def row_inputs(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Model inputs of one row. DPO rows use 'prompt'; benches use a text field."""
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        out = [{"role": m.get("role"), "content": m.get("content")} for m in msgs
               if isinstance(m, dict) and m.get("role") in INPUT_ROLES and isinstance(m.get("content"), str)]
        if out:
            return out
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return [{"role": m.get("role"), "content": m.get("content")} for m in prompt
                if isinstance(m, dict) and m.get("role") in INPUT_ROLES and isinstance(m.get("content"), str)]
    for field in ("prompt",) + TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return [{"role": "user", "content": value}]
        if isinstance(value, dict):
            return [{"role": "user", "content": json.dumps(value, ensure_ascii=False, sort_keys=True)}]
    return []


def fingerprints(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    inputs = row_inputs(row)
    if not inputs:
        return None
    users = [m["content"] for m in inputs if m["role"] == "user"]
    last_user = normalize(users[-1]) if users else ""
    return {
        "exact": _sha([{"role": m["role"], "content": _nfc(m["content"])} for m in inputs]),
        "norm_full": _sha([[m["role"], normalize(m["content"])] for m in inputs]),
        "norm_user": hashlib.sha256(last_user.encode()).hexdigest() if last_user else None,
        "user_text": last_user,
    }


def group_key(row: Dict[str, Any], fields: Sequence[str]) -> Optional[str]:
    vals = [row.get(f) for f in fields]
    if all(v not in (None, "") for v in vals):
        return "|".join(str(v) for v in vals)
    return None


def read_jsonl(path: str) -> Tuple[List[Dict[str, Any]], List[str], Optional[str]]:
    errors: List[str] = []
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        return [], [f"{path}: unreadable ({type(exc).__name__})"], None
    rows: List[Dict[str, Any]] = []
    text = raw.decode("utf-8", errors="strict") if raw else ""
    if path.endswith(".json"):
        try:
            data = json.loads(text)
            rows = data if isinstance(data, list) else data.get("rows") or data.get("data") or []
        except (json.JSONDecodeError, AttributeError) as exc:
            errors.append(f"{path}: invalid JSON ({exc.__class__.__name__})")
    else:
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{path}: line {n} invalid JSON")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                errors.append(f"{path}: line {n} not an object")
    return rows, errors, hashlib.sha256(raw).hexdigest()


def shingles(text: str, k: int = 5) -> set:
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def near_pairs(a: List[str], b: List[str], threshold: float, k: int = 5) -> List[Tuple[int, int, float]]:
    """Brute force with an inverted index; fine for corpora up to ~1e5 rows."""
    sb = [shingles(t, k) for t in b]
    index: Dict[str, List[int]] = defaultdict(list)
    for j, s in enumerate(sb):
        for g in s:
            index[g].append(j)
    out: List[Tuple[int, int, float]] = []
    for i, t in enumerate(a):
        sa = shingles(t, k)
        if not sa:
            continue
        counts: Dict[int, int] = defaultdict(int)
        for g in sa:
            for j in index.get(g, ()):
                counts[j] += 1
        for j, inter in counts.items():
            union = len(sa) + len(sb[j]) - inter
            jac = inter / union if union else 0.0
            if jac >= threshold:
                out.append((i, j, round(jac, 4)))
    return out


ID_FIELDS = ("case_id", "id", "problem_id", "task_id")


def _content_text(row: Dict[str, Any]) -> str:
    """Non-system message text, or bench text fields when a row has no messages."""
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return "\n".join(m.get("content") for m in msgs if isinstance(m, dict) and m.get("role") != "system"
                         and isinstance(m.get("content"), str))
    parts: List[str] = []
    for key in TEXT_FIELDS + ("expected", "answer", "output", "chosen", "rejected"):
        value = row.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (dict, list)):
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def word_grams(text: str, n: int = 8) -> set:
    words = normalize(text).split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def transformed_copy(train_rows: List[Dict[str, Any]], ref_rows: List[Dict[str, Any]], *, n: int = 8,
                     boilerplate_fraction: float = 0.5, containment: float = 0.5,
                     min_grams: int = 5) -> Dict[str, Any]:
    """Flag reference rows whose distinctive 8-grams are mostly present in train.

    Grams that occur in more than ``boilerplate_fraction`` of train rows or of
    reference rows are template text and are ignored. A reference row is a
    transformed copy when at least ``containment`` of its remaining grams (and at
    least ``min_grams`` of them) occur in train text.
    """
    # ids are searched in train content and non-identifier fields, never in a train row's own id field
    joined = "\n".join(json.dumps({k: v for k, v in r.items() if k not in ID_FIELDS}, ensure_ascii=False)
                       for r in train_rows)
    ids = {str(r.get(f)) for r in ref_rows for f in ID_FIELDS
           if isinstance(r.get(f), (str, int)) and len(str(r.get(f))) >= 4}
    leaked = sorted(i for i in ids if i in joined)
    def df(rows):
        counts: Dict[str, int] = defaultdict(int)
        grams = [word_grams(_content_text(r), n) for r in rows]
        for g in grams:
            for gram in g:
                counts[gram] += 1
        return grams, counts
    train_grams, tdf = df(train_rows)
    ref_grams, rdf = df(ref_rows)
    tlimit = boilerplate_fraction * max(1, len(train_rows))
    rlimit = max(1.0, boilerplate_fraction * len(ref_rows))
    vocab = {gram for gram, c in tdf.items() if c <= tlimit}
    flagged = 0
    for g in ref_grams:
        distinct = {gram for gram in g if rdf[gram] <= rlimit and tdf.get(gram, 0) <= tlimit}
        if len(distinct) >= min_grams and len(distinct & vocab) / len(distinct) >= containment:
            flagged += 1
    return {"id_leak_ids": len(leaked), "ngram_ref_rows": flagged, "ngram_n": n,
            "containment": containment}


def compare(train_rows: List[Dict[str, Any]], ref_rows: List[Dict[str, Any]], *, near_threshold: float,
            near: bool = True) -> Dict[str, Any]:
    tf = [fingerprints(r) for r in train_rows]
    rf = [fingerprints(r) for r in ref_rows]
    res: Dict[str, Any] = {"train_rows": len(train_rows), "ref_rows": len(ref_rows),
                           "train_rows_without_inputs": sum(f is None for f in tf),
                           "ref_rows_without_inputs": sum(f is None for f in rf)}
    for level in ("exact", "norm_full", "norm_user"):
        rset = {f[level] for f in rf if f and f[level]}
        hits = [i for i, f in enumerate(tf) if f and f[level] and f[level] in rset]
        refhits = {f[level] for f in rf if f and f[level]} & {f[level] for f in tf if f and f[level]}
        res[level] = {"train_rows_matching": len(hits),
                      "ref_rows_matched": sum(1 for f in rf if f and f[level] in refhits)}
    if near:
        ta = [f["user_text"] if f else "" for f in tf]
        rb = [f["user_text"] if f else "" for f in rf]
        pairs = near_pairs(ta, rb, near_threshold)
        res["near"] = {"threshold": near_threshold, "pairs": len(pairs),
                       "train_rows_matching": len({i for i, _, _ in pairs}),
                       "ref_rows_matched": len({j for _, j, _ in pairs})}
        res["transformed"] = transformed_copy(train_rows, ref_rows)
    return res


def check_manifest(manifest_path: Optional[str], holdout_path: str, holdout_sha: Optional[str],
                   holdout_rows: int) -> List[str]:
    if not manifest_path:
        return ["GATE_HOLDOUT_UNSEALED: no sealed holdout manifest"]
    try:
        m = json.loads(Path(manifest_path).read_text())
    except (OSError, json.JSONDecodeError):
        return ["GATE_HOLDOUT_MANIFEST_UNREADABLE"]
    codes = []
    if m.get("schema") != MANIFEST_SCHEMA:
        codes.append("GATE_HOLDOUT_MANIFEST_SCHEMA")
    if m.get("sha256") != holdout_sha:
        codes.append("GATE_HOLDOUT_SHA_MISMATCH")
    if m.get("rows") != holdout_rows:
        codes.append("GATE_HOLDOUT_ROWCOUNT_MISMATCH")
    if not m.get("frozen_at"):
        codes.append("GATE_HOLDOUT_NOT_FROZEN")
    return codes


def run_gate(train: str, holdout: str, manifest: Optional[str], references: Sequence[str], *,
             near_threshold: float = 0.8, eval_dataset_ids: Sequence[str] = (), group_fields: Sequence[str] = ("source_dataset", "source_revision"),
             fallback_group_field: str = "source_row_sha256") -> Dict[str, Any]:
    codes: List[str] = []
    errors: List[str] = []
    train_rows, e, train_sha = read_jsonl(train)
    errors += e
    hold_rows, e, hold_sha = read_jsonl(holdout)
    errors += e
    if errors or train_sha is None or hold_sha is None:
        codes.append("GATE_INPUT_INVALID")
    if not train_rows:
        codes.append("GATE_TRAIN_EMPTY")
    if not hold_rows:
        codes.append("GATE_HOLDOUT_EMPTY")
    codes += check_manifest(manifest, holdout, hold_sha, len(hold_rows))
    report: Dict[str, Any] = {"schema": SCHEMA, "train": {"path": train, "sha256": train_sha, "rows": len(train_rows)},
                              "holdout": {"path": holdout, "sha256": hold_sha, "rows": len(hold_rows)}}
    # self-duplicates inside train (informational) and inside holdout (must be 0)
    hold = compare(train_rows, hold_rows, near_threshold=near_threshold)
    report["vs_holdout"] = hold
    if hold["exact"]["train_rows_matching"] or hold["norm_full"]["train_rows_matching"] \
            or hold["norm_user"]["train_rows_matching"]:
        codes.append("GATE_HOLDOUT_OVERLAP")
    if hold.get("near", {}).get("pairs"):
        codes.append("GATE_HOLDOUT_NEAR_DUPLICATE")
    if hold.get("transformed", {}).get("id_leak_ids") or hold.get("transformed", {}).get("ngram_ref_rows"):
        codes.append("GATE_HOLDOUT_TRANSFORMED_COPY")
    banned = {d.casefold() for d in eval_dataset_ids}
    from_eval = sum(1 for r in train_rows if str(r.get("source_dataset", "")).casefold() in banned)
    report["train_rows_from_eval_datasets"] = from_eval
    if from_eval:
        codes.append("GATE_TRAIN_FROM_EVAL_SOURCE")
    if hold["train_rows_without_inputs"] or hold["ref_rows_without_inputs"]:
        codes.append("GATE_ROWS_WITHOUT_INPUTS")

    def groups(rows):
        g = {group_key(r, group_fields) for r in rows} - {None}
        return g or ({group_key(r, (fallback_group_field,)) for r in rows} - {None})
    tg, hg = groups(train_rows), groups(hold_rows)
    report["groups"] = {"train": len(tg), "holdout": len(hg), "overlap": len(tg & hg)}
    if tg & hg:
        codes.append("GATE_GROUP_OVERLAP")
    report["references"] = []
    for ref in references:
        ref_rows, e, ref_sha = read_jsonl(ref)
        if e or ref_sha is None or not ref_rows:
            codes.append("GATE_REFERENCE_INVALID")
            report["references"].append({"path": ref, "status": "INVALID", "errors": e[:5]})
            continue
        res = compare(train_rows, ref_rows, near_threshold=near_threshold)
        res.update({"path": ref, "sha256": ref_sha})
        report["references"].append(res)
        if res["exact"]["train_rows_matching"] or res["norm_full"]["train_rows_matching"] \
                or res["norm_user"]["train_rows_matching"]:
            codes.append("GATE_REFERENCE_OVERLAP")
        if res.get("near", {}).get("pairs"):
            codes.append("GATE_REFERENCE_NEAR_DUPLICATE")
        if res.get("transformed", {}).get("id_leak_ids") or res.get("transformed", {}).get("ngram_ref_rows"):
            codes.append("GATE_REFERENCE_TRANSFORMED_COPY")
    seen = []
    for c in codes:
        if c not in seen:
            seen.append(c)
    report.update({"codes": seen, "errors": errors[:25], "ok": not seen,
                   "verdict": "CLEAN_NOT_LAUNCHED" if not seen else "REJECT",
                   "limits": "lexical matching only; paraphrase/semantic leakage is not detected"})
    return report


def seal(holdout: str, out: str, frozen_at: str, group_fields: Sequence[str] = ("source_dataset", "source_revision")) -> Dict[str, Any]:
    rows, errors, sha = read_jsonl(holdout)
    if errors or sha is None or not rows:
        raise SystemExit(f"cannot seal: {errors[:3] or 'empty'}")
    m = {"schema": MANIFEST_SCHEMA, "path": holdout, "sha256": sha, "rows": len(rows), "frozen_at": frozen_at,
         "groups": sorted({group_key(r, group_fields) for r in rows} - {None}),
         "input_exact_fingerprints": sorted({f["exact"] for f in map(fingerprints, rows) if f})}
    Path(out).write_text(json.dumps(m, indent=1, sort_keys=True))
    return m


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("check")
    g.add_argument("--train", required=True)
    g.add_argument("--holdout", required=True)
    g.add_argument("--manifest")
    g.add_argument("--reference", action="append", default=[])
    g.add_argument("--near-threshold", type=float, default=0.8)
    g.add_argument("--eval-dataset-id", action="append", default=[])
    s = sub.add_parser("seal")
    s.add_argument("--holdout", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--frozen-at", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "seal":
        print(json.dumps(seal(a.holdout, a.out, a.frozen_at), indent=1)[:2000])
        return 0
    rep = run_gate(a.train, a.holdout, a.manifest, a.reference, near_threshold=a.near_threshold,
                   eval_dataset_ids=a.eval_dataset_id)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
