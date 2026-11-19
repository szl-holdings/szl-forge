#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decontamination extension (LOCAL PROPOSAL; stdlib only; reads files, writes nothing but a report).

Adds to the A21 gate (triage6/patches/tools/contamination_gate.py) three things the leaders do:
  1. sampled word n-gram detection with bidirectional cluster expansion and an IDF-weighted
     contamination score (the design of allenai/decon, used for Olmo 3 midtraining);
  2. MinHash+LSH near-duplicate detection with exact-Jaccard verification (datatrove / NeMo Curator fuzzy dedup);
  3. group-disjoint checks and an all-pairs cross-campaign leak matrix (the W29 failure).
Lexical only: it does not detect paraphrase, translation or semantic leakage.
A clean report is CLEAN_NOT_LAUNCHED. It never authorizes a launch.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, re, sys, unicodedata
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

_ZW = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def words(text: str) -> List[str]:
    t = unicodedata.normalize("NFKC", text)
    t = _ZW.sub("", t).casefold()
    t = _PUNCT.sub(" ", t)
    return t.split()


def row_prompt(r: dict, include_assistant: bool = False) -> str:
    if isinstance(r.get("messages"), list):
        return "\n".join(str(m.get("content", "")) for m in r["messages"]
                         if include_assistant or m.get("role") in ("system", "user"))
    for k in ("text", "prompt", "input", "question", "problem", "instruction", "scenario"):
        if k in r:
            return str(r[k])
    return json.dumps(r, sort_keys=True)


def _h(tokens: Sequence[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(tokens).encode(), digest_size=8).digest(), "big")


# ---------------------------------------------------------------- 1. sampled n-gram + cluster expansion
class NgramIndex:
    """Inverted index of reference (eval/holdout) word n-grams -> reference doc ids, with IDF weights."""

    def __init__(self, refs: Sequence[str], n: int = 5):
        self.n = n
        self.docs: List[List[int]] = []          # per ref doc: ordered n-gram hashes
        self.inv: Dict[int, Set[int]] = defaultdict(set)
        for i, t in enumerate(refs):
            w = words(t)
            g = [_h(w[j:j + n]) for j in range(len(w) - n + 1)] if len(w) >= n else []
            self.docs.append(g)
            for x in g:
                self.inv[x].add(i)
        N = max(len(refs), 1)
        self.idf = {x: math.log(1 + N / len(s)) for x, s in self.inv.items()}
        self.short = {i: " ".join(words(t)) for i, t in enumerate(refs) if len(words(t)) < n and words(t)}


def detect(index: NgramIndex, train_texts: Sequence[str], sample_every: int = 3, max_misses: int = 4,
           threshold: float = 0.8, min_ref_ngrams: int = 3) -> List[dict]:
    """Return one hit per (train doc, ref doc) with idf-weighted coverage >= threshold.
    Sampling finds candidate clusters cheaply; each cluster is then expanded exhaustively left and
    right, tolerating up to `max_misses` consecutive non-matching n-grams (edits/insertions)."""
    n = index.n
    hits: Dict[Tuple[int, int], dict] = {}
    for ti, t in enumerate(train_texts):
        w = words(t)
        if len(w) >= n:
            g = [_h(w[j:j + n]) for j in range(len(w) - n + 1)]
            seen_start: Set[int] = set()
            for pos in range(0, len(g), max(1, sample_every)):
                for rid in index.inv.get(g[pos], ()):
                    if (ti, rid, pos) in seen_start:
                        continue
                    matched: Set[int] = set()
                    # expand both directions over positions; a ref ngram counts once
                    refset = set(index.docs[rid])
                    for direction in (1, -1):
                        p, miss = pos, 0
                        while 0 <= p < len(g) and miss <= max_misses:
                            if g[p] in refset:
                                matched.add(g[p]); miss = 0
                            else:
                                miss += 1
                            p += direction
                    ref_total = set(index.docs[rid])
                    if len(ref_total) < min_ref_ngrams:
                        continue
                    num = sum(index.idf[x] for x in matched)
                    den = sum(index.idf[x] for x in ref_total)
                    score = num / den if den else 0.0
                    key = (ti, rid)
                    if score >= threshold and (key not in hits or score > hits[key]["score"]):
                        hits[key] = {"train_index": ti, "ref_index": rid, "score": round(score, 4),
                                     "matched_ngrams": len(matched), "ref_ngrams": len(ref_total), "kind": "ngram_cluster"}
        # short references (< n words): exact normalized containment as a whole phrase
        if index.short:
            joined = " ".join(w)
            for rid, phrase in index.short.items():
                if len(phrase) >= 12 and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", joined):
                    hits[(ti, rid)] = {"train_index": ti, "ref_index": rid, "score": 1.0,
                                       "matched_ngrams": 0, "ref_ngrams": 0, "kind": "short_phrase_exact"}
    return sorted(hits.values(), key=lambda h: (h["train_index"], h["ref_index"]))


# ---------------------------------------------------------------- 2. MinHash + LSH
class MinHasher:
    P = (1 << 61) - 1

    def __init__(self, num_perm: int = 128, seed: int = 1337, shingle: int = 3):
        rnd = random.Random(seed)
        self.a = [rnd.randrange(1, self.P) for _ in range(num_perm)]
        self.b = [rnd.randrange(0, self.P) for _ in range(num_perm)]
        self.k = shingle
        self.num_perm = num_perm

    def shingles(self, text: str) -> Set[int]:
        w = words(text)
        if len(w) < self.k:
            return {_h(w)} if w else set()
        return {_h(w[i:i + self.k]) for i in range(len(w) - self.k + 1)}

    def signature(self, text: str) -> Tuple[int, ...]:
        sh = self.shingles(text)
        if not sh:
            return tuple([self.P] * self.num_perm)
        return tuple(min((a * x + b) % self.P for x in sh) for a, b in zip(self.a, self.b))


def jaccard(a: Set[int], b: Set[int]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def near_duplicates(left: Sequence[str], right: Sequence[str], threshold: float = 0.8,
                    num_perm: int = 128, bands: int = 32, seed: int = 1337) -> List[dict]:
    """Pairs (i in left, j in right) with exact word-shingle Jaccard >= threshold, found via LSH then verified."""
    assert num_perm % bands == 0
    rows = num_perm // bands
    mh = MinHasher(num_perm, seed)
    sigs_r = [mh.signature(t) for t in right]
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
    for j, s in enumerate(sigs_r):
        for b in range(bands):
            buckets[(b, s[b * rows:(b + 1) * rows])].append(j)
    out, seen = [], set()
    for i, t in enumerate(left):
        s = mh.signature(t)
        cand: Set[int] = set()
        for b in range(bands):
            cand.update(buckets.get((b, s[b * rows:(b + 1) * rows]), ()))
        sh_i = mh.shingles(t)
        for j in sorted(cand):
            if (i, j) in seen:
                continue
            seen.add((i, j))
            jac = jaccard(sh_i, mh.shingles(right[j]))
            if jac >= threshold:
                out.append({"left_index": i, "right_index": j, "jaccard": round(jac, 4), "kind": "minhash_near_dup"})
    return out


def strip_boilerplate(texts: Sequence[str], n: int = 6, frac: float = 0.3) -> List[str]:
    """Remove words covered by word n-grams that occur in >= frac of the documents (shared persona/system/template text),
    so overlap is judged on the distinctive part of each prompt. Returns normalized word strings."""
    toks = [words(t) for t in texts]
    df: Dict[int, int] = defaultdict(int)
    for w in toks:
        for g in {_h(w[i:i + n]) for i in range(len(w) - n + 1)}:
            df[g] += 1
    cut = max(2, int(frac * len(texts)))
    out = []
    for w in toks:
        drop = set()
        for i in range(len(w) - n + 1):
            if df[_h(w[i:i + n])] >= cut:
                drop.update(range(i, i + n))
        out.append(" ".join(x for j, x in enumerate(w) if j not in drop))
    return out


# ---------------------------------------------------------------- 3. groups and cross-campaign matrix
def default_group(r: dict) -> Optional[str]:
    for k in ("group", "source_group"):
        if r.get(k):
            return str(r[k])
    if r.get("source_dataset"):
        return f"{r['source_dataset']}@{r.get('source_revision', '')}"
    if r.get("source_row_sha256"):
        return str(r["source_row_sha256"])
    return None


def group_overlap(train: Sequence[dict], holdout: Sequence[dict],
                  key: Callable[[dict], Optional[str]] = default_group) -> dict:
    tg = {key(r) for r in train} - {None}
    hg = [key(r) for r in holdout]
    missing = sum(1 for g in hg if g is None)
    shared = [i for i, g in enumerate(hg) if g is not None and g in tg]
    return {"holdout_rows": len(holdout), "rows_without_group_key": missing, "holdout_rows_sharing_group_with_train": len(shared),
            "group_disjoint": (len(shared) == 0 and missing == 0)}


def cross_matrix(campaigns: Dict[str, Dict[str, Sequence[dict]]], threshold: float = 0.8) -> Dict[str, Dict[str, dict]]:
    """campaigns[name] = {"train": rows, "holdout": rows}. For every ordered pair (A,B) with A != B:
    how many of B's holdout rows are exact/near/ngram-covered by A's train rows."""
    res: Dict[str, Dict[str, dict]] = {}
    for a, ca in campaigns.items():
        tt = [row_prompt(r) for r in ca["train"]]
        exact = {" ".join(words(x)) for x in tt}
        res[a] = {}
        for b, cb in campaigns.items():
            if a == b:
                continue
            ht = [row_prompt(r) for r in cb["holdout"]]
            ex = sum(1 for x in ht if " ".join(words(x)) in exact)
            idx = NgramIndex(ht, n=5)
            ng = len({h["ref_index"] for h in detect(idx, tt, sample_every=2, threshold=threshold)})
            nd = len({h["right_index"] for h in near_duplicates(tt, ht, threshold)})
            res[a][b] = {"holdout_rows": len(ht), "exact": ex, "ngram_cluster": ng, "near_dup": nd}
    return res


# ---------------------------------------------------------------- report / CLI
def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(train: List[dict], holdout: List[dict], threshold: float = 0.8, group_key=default_group,
        boilerplate: bool = True) -> dict:
    tt = [row_prompt(r) for r in train]
    ht = [row_prompt(r) for r in holdout]
    if boilerplate and len(tt) >= 10:
        both = strip_boilerplate(tt + ht)
        tt, ht = both[:len(tt)], both[len(tt):]
    ng = detect(NgramIndex(ht, 5), tt, threshold=threshold)
    nd = near_duplicates(tt, ht, threshold)
    grp = group_overlap(train, holdout, group_key)
    contaminated = {h["ref_index"] for h in ng} | {h["right_index"] for h in nd}
    verdict = "REJECT" if (contaminated or not grp["group_disjoint"]) else "CLEAN_NOT_LAUNCHED"
    return {"schema": "szl.forge.decon-ext/v1", "verdict": verdict, "train_rows": len(train), "holdout_rows": len(holdout),
            "holdout_rows_contaminated": len(contaminated), "ngram_hits": len(ng), "near_dup_hits": len(nd),
            "group": grp, "boilerplate_stripped": bool(boilerplate and len(train) >= 10), "limits": "lexical only; no paraphrase/semantic detection"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True); ap.add_argument("--holdout", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    a = ap.parse_args(argv)
    rep = run(load_jsonl(a.train), load_jsonl(a.holdout), a.threshold)
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] == "CLEAN_NOT_LAUNCHED" else 1


if __name__ == "__main__":
    sys.exit(main())
