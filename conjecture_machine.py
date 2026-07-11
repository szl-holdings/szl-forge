#!/usr/bin/env python3
# SZL Conjecture Machine — point the sovereign stack at the formula corpus.
#
# WHAT IT DOES
#   Loads a formula index (ships with a local snapshot of the estate's
#   thesis-formula-index), iterates each formula, and asks the sovereign
#   OpenAI-compatible endpoint (own metal — the tower / laptop / nemo slot)
#   for a PROOF SKETCH, a LEMMA DECOMPOSITION, and a COUNTEREXAMPLE SEARCH.
#   Every attempt is written to conjecture_runs/<ts>/<formula_id>.json.
#
# HONESTY DOCTRINE (hard-coded, non-negotiable — see szl-forge README):
#   - This machine NEVER claims a formula is "proven". A model sketch is
#     ADVISORY only. A formula's status stays CONJECTURE forever unless a real
#     Lean check passes (lean_check() below is an honest stub — no Lean
#     toolchain is wired here, so it never passes; the dataset's own "GREEN"
#     label is NOT treated as a live proof by this machine).
#   - Λ (governance trust quantity) uniqueness is Conjecture-1, permanently.
#     A hard guard forces any Λ-uniqueness formula to CONJECTURE and refuses
#     to ever upgrade it, even if a Lean check were later supplied.
#   - Endpoint down => record honest UNAVAILABLE and exit cleanly. Never
#     fabricate model output.
#
# DEPENDENCIES: Python 3.8+ standard library only (urllib). No pip installs.
#
# USAGE (one command; see RUNBOOK-CONJECTURE.md for the owner walk-through):
#   python conjecture_machine.py                 # live run vs the tower
#   python conjecture_machine.py --dry-run       # no network, prove wiring
#   python conjecture_machine.py --limit 5       # first 5 formulas only
#   MODEL=szl-nemo python conjecture_machine.py  # use the nemo slot

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "https://gpu.a-11-oy.com/v1"
DEFAULT_MODEL = "llama3-szl-finetuned-q4:latest"  # or szl-nemo (env MODEL / --model)
DEFAULT_CORPUS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "thesis_formula_index.json"
)

# The permanent doctrine line. Echoed into every run manifest.
DOCTRINE = (
    "This machine NEVER claims proven. Model output is an advisory sketch only. "
    "Status stays CONJECTURE unless a real Lean check passes. "
    "Lambda uniqueness is Conjecture-1, permanently non-upgradable."
)

# Patterns that identify the Lambda-uniqueness conjecture (Conjecture-1). Any
# formula matching these is doctrine-locked to CONJECTURE forever.
LAMBDA_LOCK_PATTERNS = [
    re.compile(r"\blambda\b.*\buniqu", re.IGNORECASE),
    re.compile(r"\buniqu.*\blambda\b", re.IGNORECASE),
    re.compile(r"\bconjecture[\s\-_]?1\b", re.IGNORECASE),
    re.compile(r"\bTH[_\-]?L1\b", re.IGNORECASE),
    re.compile(r"\bLambda[\s\-_]?uniqueness\b", re.IGNORECASE),
    re.compile(r"Λ.*uniqu", re.IGNORECASE),
]


def is_lambda_locked(formula):
    """True if this formula is the Lambda-uniqueness conjecture (Conjecture-1)."""
    haystack = " ".join(
        str(formula.get(k, ""))
        for k in ("id", "title", "description", "statement", "lean_theorem")
    )
    return any(p.search(haystack) for p in LAMBDA_LOCK_PATTERNS)


def load_corpus(path):
    """Load a formula index and normalize to a list of dicts.

    Supports: {"meta":..., "entries":[...]} (thesis-formula-index),
    {"formulas":[...]} / {"formulas":{...}}, or a bare list. Counts the ACTUAL
    entries, never a possibly-stale meta.entry_count.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = {}
    if isinstance(data, dict):
        meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
        if isinstance(data.get("entries"), list):
            entries = data["entries"]
        elif isinstance(data.get("formulas"), list):
            entries = data["formulas"]
        elif isinstance(data.get("formulas"), dict):
            entries = list(data["formulas"].values())
        else:
            # dict-of-formulas fallback (skip a top-level "meta" key)
            entries = [v for k, v in data.items() if k != "meta" and isinstance(v, dict)]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Unrecognized corpus shape: expected dict or list")
    return meta, [e for e in entries if isinstance(e, dict)]


def formula_id(formula, idx):
    for k in ("id", "formula_id", "name", "key"):
        v = formula.get(k)
        if v:
            return re.sub(r"[^A-Za-z0-9._-]+", "_", str(v))
    return "formula_%03d" % idx


def formula_statement(formula):
    for k in ("description", "statement", "latex", "title", "claim"):
        v = formula.get(k)
        if v:
            return str(v)
    return "(no statement text in corpus entry)"


def build_prompt(formula):
    title = formula.get("title", "")
    section = formula.get("thesis_section", "")
    statement = formula_statement(formula)
    lean_theorem = formula.get("lean_theorem", "")
    system = (
        "You are a careful mathematical assistant operating under the SZL honesty "
        "doctrine. You produce ADVISORY analysis only: proof SKETCHES, lemma "
        "decompositions, and counterexample searches. You NEVER assert that "
        "something is proven — a machine-checked Lean proof is the only proof that "
        "counts, and that is done elsewhere. If you are unsure, say UNKNOWN. Do not "
        "fabricate citations or numbers."
    )
    user = (
        "Formula/claim under study"
        + (" (%s)" % section if section else "")
        + ":\n"
        + (("Title: %s\n" % title) if title else "")
        + ("Statement: %s\n" % statement)
        + (("Referenced Lean theorem: %s\n" % lean_theorem) if lean_theorem else "")
        + "\nProvide, as advisory analysis only:\n"
        + "1. PROOF SKETCH — the shape of an argument (no claim of proof).\n"
        + "2. LEMMA DECOMPOSITION — sub-lemmas a formal proof would need.\n"
        + "3. COUNTEREXAMPLE SEARCH — edge cases / regimes where it might fail, "
        + "or 'none found' with why.\n"
        + "End with one line: CONFIDENCE: <low|medium|high> that a formal proof exists."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_model(endpoint, model, messages, token, timeout, cf_id, cf_secret):
    """POST /chat/completions. Returns (text, latency_ms, error)."""
    url = endpoint.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 900,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer %s" % (token or "sovereign-local"),
    }
    if cf_id and cf_secret:
        headers["CF-Access-Client-Id"] = cf_id
        headers["CF-Access-Client-Secret"] = cf_secret
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        latency = int((time.time() - started) * 1000)
        text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not text:
            return None, latency, "empty completion from endpoint"
        return text, latency, None
    except urllib.error.HTTPError as exc:
        latency = int((time.time() - started) * 1000)
        return None, latency, "HTTP %s from endpoint" % exc.code
    except Exception as exc:  # noqa: BLE001 — honest catch-all, network is flaky
        latency = int((time.time() - started) * 1000)
        return None, latency, "%s: %s" % (type(exc).__name__, exc)


def lean_check(formula):
    """Honest stub. No Lean toolchain is wired here, so this NEVER passes.

    A real check would `lake build` the referenced theorem in
    szl-holdings/lutar-lean and confirm zero `sorry`. Until that is wired, every
    formula stays CONJECTURE — the dataset's own 'GREEN'/'status' label is a
    STAGED-ADVISORY tag, NOT a live proof, and this machine does not launder it
    into 'proven'.
    """
    return {
        "attempted": False,
        "passed": False,
        "reason": (
            "Lean checking not wired in this environment; would require "
            "`lake build` on szl-holdings/lutar-lean. Dataset status label is "
            "advisory, not a live proof."
        ),
        "referenced_lean_theorem": formula.get("lean_theorem") or None,
        "referenced_lean_file": formula.get("lean_file") or None,
    }


def health_check(endpoint, token, timeout, cf_id, cf_secret):
    """GET /models. Returns (up: bool, detail: str)."""
    url = endpoint.rstrip("/") + "/models"
    headers = {"Authorization": "Bearer %s" % (token or "sovereign-local")}
    if cf_id and cf_secret:
        headers["CF-Access-Client-Id"] = cf_id
        headers["CF-Access-Client-Secret"] = cf_secret
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, "HTTP %s" % resp.status
    except urllib.error.HTTPError as exc:
        return False, "HTTP %s" % exc.code
    except Exception as exc:  # noqa: BLE001
        return False, "%s: %s" % (type(exc).__name__, exc)


def record_for(formula, idx, status, model_output, error, latency, endpoint, model):
    locked = is_lambda_locked(formula)
    lean = lean_check(formula)
    # Doctrine guards: proven is ALWAYS false; Λ-uniqueness is hard-locked.
    if locked:
        status = "CONJECTURE"
    return {
        "formula_id": formula_id(formula, idx),
        "title": formula.get("title") or None,
        "thesis_section": formula.get("thesis_section") or None,
        "statement": formula_statement(formula),
        "endpoint": endpoint,
        "model": model,
        "requested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": status,  # CONJECTURE | UNAVAILABLE | DRY_RUN
        "proven": False,  # doctrine: this machine never claims proven
        "lean_check": lean,
        "lambda_uniqueness_locked": locked,
        "doctrine_note": (
            "Λ uniqueness is Conjecture-1 — permanently non-provable by this machine."
            if locked
            else "Advisory sketch only; not a proof. Status stays CONJECTURE until a Lean check passes."
        ),
        "model_output": model_output,  # advisory sketch, or null
        "error": error,
        "latency_ms": latency,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SZL Conjecture Machine — sovereign stack x formula corpus."
    )
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS, help="path to formula index JSON"
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("A11OY_MODEL_BASE_URL") or DEFAULT_ENDPOINT,
        help="OpenAI-compatible base URL of the sovereign endpoint",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL")
        or os.environ.get("SOVEREIGN_MODEL")
        or DEFAULT_MODEL,
        help="model name (e.g. llama3-szl-finetuned-q4:latest or szl-nemo)",
    )
    parser.add_argument(
        "--token", default=os.environ.get("A11OY_GPU_TOKEN", "sovereign-local")
    )
    parser.add_argument("--out", default="conjecture_runs", help="output directory")
    parser.add_argument("--limit", type=int, default=0, help="only first N formulas")
    parser.add_argument("--timeout", type=int, default=120, help="per-call timeout (s)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="no network calls; write DRY_RUN records to prove wiring",
    )
    args = parser.parse_args(argv)

    cf_id = os.environ.get("A11OY_GPU_CF_ACCESS_ID") or None
    cf_secret = os.environ.get("A11OY_GPU_CF_ACCESS_SECRET") or None

    try:
        meta, formulas = load_corpus(args.corpus)
    except Exception as exc:  # noqa: BLE001
        print("[conjecture] ERROR loading corpus: %s" % exc, file=sys.stderr)
        return 2

    if args.limit and args.limit > 0:
        formulas = formulas[: args.limit]

    total = len(formulas)
    declared = meta.get("entry_count")
    print("[conjecture] corpus: %s" % args.corpus)
    print("[conjecture] formulas (actual count): %d" % total)
    if declared is not None and declared != total:
        print(
            "[conjecture] NOTE: corpus meta.entry_count=%s disagrees with actual %d "
            "(reporting the actual count)." % (declared, total)
        )

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out, ts)
    os.makedirs(run_dir, exist_ok=True)

    endpoint_up = False
    endpoint_detail = "dry-run (no probe)"
    if not args.dry_run:
        endpoint_up, endpoint_detail = health_check(
            args.endpoint, args.token, min(args.timeout, 30), cf_id, cf_secret
        )
        print(
            "[conjecture] endpoint %s -> %s (%s)"
            % (
                args.endpoint,
                "UP" if endpoint_up else "UNAVAILABLE",
                endpoint_detail,
            )
        )
        if not endpoint_up:
            print(
                "[conjecture] endpoint UNAVAILABLE — recording honest UNAVAILABLE per "
                "formula, no model output fabricated."
            )

    tally = {"CONJECTURE": 0, "UNAVAILABLE": 0, "DRY_RUN": 0}
    for idx, formula in enumerate(formulas):
        fid = formula_id(formula, idx)
        if args.dry_run:
            rec = record_for(
                formula, idx, "DRY_RUN", None, "dry-run: no call made",
                None, args.endpoint, args.model,
            )
        elif not endpoint_up:
            rec = record_for(
                formula, idx, "UNAVAILABLE", None, "endpoint unavailable: %s" % endpoint_detail,
                None, args.endpoint, args.model,
            )
        else:
            text, latency, error = call_model(
                args.endpoint,
                args.model,
                build_prompt(formula),
                args.token,
                args.timeout,
                cf_id,
                cf_secret,
            )
            status = "CONJECTURE" if text else "UNAVAILABLE"
            rec = record_for(
                formula, idx, status, text, error, latency, args.endpoint, args.model
            )
        tally[rec["status"]] = tally.get(rec["status"], 0) + 1
        with open(os.path.join(run_dir, "%s.json" % fid), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        print("  [%d/%d] %s -> %s" % (idx + 1, total, fid, rec["status"]))

    manifest = {
        "run_ts": ts,
        "corpus": os.path.abspath(args.corpus),
        "corpus_meta": meta or None,
        "formula_count_actual": total,
        "formula_count_declared": declared,
        "endpoint": args.endpoint,
        "endpoint_status": "UP" if endpoint_up else ("DRY_RUN" if args.dry_run else "UNAVAILABLE"),
        "endpoint_detail": endpoint_detail,
        "model": args.model,
        "dry_run": args.dry_run,
        "tally": tally,
        "doctrine": DOCTRINE,
        "proven_count": 0,  # always zero — this machine never proves anything
    }
    with open(os.path.join(run_dir, "_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print("[conjecture] wrote %d records + manifest to %s" % (total, run_dir))
    print("[conjecture] tally: %s" % tally)
    print("[conjecture] proven: 0 (by doctrine — nothing is ever claimed proven here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
