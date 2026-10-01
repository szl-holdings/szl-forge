#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# (c) 2026 SZL Holdings
"""SZL Holdings estate payload v4 - owner-metal, one-paste, stall-proof operator.

Lanes (each records OK or BLOCKED with a trace; the others keep running):
  discover     -> live GitHub org + Hugging Face org inventory, szl-forge source bindings and per-folder evidence
  analyze      -> artifact classification, scores, quantization drift, source/artifact sync, governance gaps,
                  and a per-model SOURCE MAP: which szl-forge folder trains/evaluates it (or nothing does)
  quarantine   -> hash-chained ledger; Hub quarantine PRs only when SZL_EXECUTE=1 (never direct commits, never deletes)
  pull_requests-> green-only merges with live re-check + head-SHA lock; BEHIND branches get updated
  pr_repair    -> failing/conflicting PRs: flake reruns, dependabot recreate/rebase, formatter pushes,
                  base-merge pushes, redacted repair briefs for everything that needs code judgment
  corpora      -> finds local owner corpora (JSONL) and validates their record shape with the candidate kit library
  candidates   -> writes a complete fail-closed training kit (frontier-candidate idiom) for every model with no
                  trainer anywhere; binds local corpora by digest; opens ONE pull request against szl-forge
  train        -> on this machine: qualify -> curriculum/leakage -> train (smoke|full) -> evaluate -> export -> card
  reports      -> SCORECARD.md, CODEX-v4.md, CODEX-PR-REPAIRS.md, OWNER-GPU-RUNBOOK.md, CSVs, summary.json

It is not a publisher, a scheduler, a merge queue, or a promotion authority. It never force-pushes, never bypasses
protection, never deletes, never trains on an unbound corpus, and never writes PROMOTABLE.
"""
import csv
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import posixpath
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

VERSION = "4.0.0"
GH_ORG = os.environ.get("SZL_GH_ORG", "szl-holdings")
HF_ORG = os.environ.get("SZL_HF_ORG", "SZLHOLDINGS")
TRAIN_REPO = os.environ.get("SZL_TRAINING_REPO", "szl-forge")
ROOT = pathlib.Path(os.environ.get("SZL_ROOT", "szl_out")).resolve()
EXECUTE = os.environ.get("SZL_EXECUTE") == "1"
MERGE = os.environ.get("SZL_MERGE") == "1"
REPAIR = os.environ.get("SZL_REPAIR") == "1"
COMMENT = os.environ.get("SZL_COMMENT_PRS", "1") == "1"
OPEN_PR = os.environ.get("SZL_OPEN_TRAINING_PR") == "1"
TRAIN_MODE = os.environ.get("SZL_TRAIN_MODE", "smoke").lower()
CONFIRM = os.environ.get("SZL_CONFIRM_BINDINGS") == "1"
CLONE = os.environ.get("SZL_CLONE") == "1"
KIT_DIR = pathlib.Path(os.environ.get("SZL_KIT_DIR") or (pathlib.Path(__file__).resolve().parent / "kit"))
DATE = dt.datetime.now().strftime("%Y%m%d")
RUN = ROOT / "runs" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
LEDGER = ROOT / "ledger" / "quarantine_ledger.jsonl"
REPOS = ROOT / "repos"
for _p in (RUN, LEDGER.parent, REPOS):
    _p.mkdir(parents=True, exist_ok=True)
LOGF = open(RUN / "run.log", "a", encoding="utf-8")
HF = "https://huggingface.co"
GHBIN = shutil.which("gh") or "gh"
GIT = shutil.which("git") or "git"
bn = posixpath.basename
sys.path.insert(0, str(KIT_DIR))
try:
    import candidate_lib as kit  # the same library every generated candidate ships with
except Exception as _e:  # pragma: no cover - reported as a lane block below
    kit = None
    KIT_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
else:
    KIT_IMPORT_ERROR = None

WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".npz", ".joblib", ".pkl", ".h5", ".msgpack")
NON_WEIGHT = {"training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt"}
LIB_HINTS = ("kernel", "attn", "block-kv", "lambda-gate", "formula", "invariant", "provctl", "govsign",
             "meter", "training-scripts", "-spec", "-sdk", "substrate", "blocked", "ouroboros", "maskmod", "norm")
SYN_HINTS = ("synthetic", "fixture", "placeholder", "toy-data", "toy_data", "mock-data", "dummy")
STOP = {"model", "models", "nano", "szl", "holdings", "gguf", "qwen", "qwen2", "qwen25", "qwen35",
        "instruct", "adapter", "lora", "base", "mini", "small", "large"}
GENERIC = STOP | {"card", "v1", "v2", "v3", "v4", "r1", "r2", "r3", "r4", "0", "1", "3b", "5b", "7b", "8b", "15b",
                  "merged", "abstain", "candidate", "forge", "study5", "study", "lab"}
FAMILIES = {"pcm-core/receipt-agent": ("receipt", "forge", "pcm"), "pcm-navigator/khipu": ("khipu", "navigator"),
            "pcm-router/chaski": ("chaski", "router"), "pcm-explain/willay": ("willay",),
            "pcm-perception/killinchu": ("killinchu",), "typesafe-triage": ("triage",), "a11oy": ("a11oy",)}
RECEIPT_RE = re.compile(r"(?i)(receipt|gate|eval_results|provenance|parity)[^/]*\.jsonl?$")
QRE = re.compile(r"(?i)(?<![a-z0-9])(iq\d_[a-z]+|q\d_k_[sml]|q\d_k|q\d_\d|bf16|f16|f32)(?![a-z0-9])")
GH_LINK = re.compile(r"github\.com/" + re.escape(GH_ORG) + r"/([A-Za-z0-9_.\-]+)", re.I)
HF_LINK = re.compile(r"huggingface\.co/(?:datasets/|spaces/)?(" + re.escape(HF_ORG) + r"/[A-Za-z0-9_.\-]+)", re.I)
PASS = {"SUCCESS", "NEUTRAL", "SKIPPED"}
FAIL = {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE", "STALE"}
MSG = {"MISSING_WEIGHTS": "Model configuration is present but no weight files were found.",
       "PLACEHOLDER_WEIGHTS": "All tensor files are under 1 MB and appear to be placeholders.",
       "CARD_ONLY": "Only a model card or metadata is present; no model artifact exists.",
       "EMPTY": "Repository contains no files.",
       "SYNTHETIC_OR_FIXTURE": "Dataset is synthetic or a test fixture; excluded from production training.",
       "LICENSE_UNKNOWN": "Dataset declares no license; excluded from production training until rights are documented."}
REPAIR_RESULTS = {"FAILING_CHECKS", "CONFLICT_NEEDS_REBASE", "MERGE_FAILED", "UPDATE_BRANCH_FAILED", "NOT_MERGEABLE_BLOCKED"}
NON_LLM_TAGS = {"not-a-model", "org-stub", "not-a-checkpoint", "source-pointer", "no-implementation-here", "logistic-regression",
                "standard-library", "kernel", "kernels", "tabular-classification", "tabular-regression", "embedding-table", "alias",
                "not-a-transformers-config", "sklearn", "scikit-learn"}
NON_LLM_LIBS = {"kernels", "sklearn", "scikit-learn", "joblib", "onnx", "other", "keras", "tf-keras", "spacy", "fasttext"}
LLM_PIPES = {"text-generation", "text2text-generation", "image-text-to-text", "text-classification", "token-classification",
             "zero-shot-classification", "summarization", "question-answering", "feature-extraction", "sentence-similarity"}
LLM_LIBS = {"peft", "transformers", "unsloth", "gguf", "trl", "sentence-transformers", "mlx", "llama.cpp"}
CLASSES = [("INTENTIONAL_RED", r"verify-proof|lambda-bounty|lambda_unique_|Conjecture 1 .*bounty"),
           ("INFRA_FLAKE", r"rate limit|ECONNRESET|ETIMEDOUT|Could not resolve host|503 Service|502 Bad Gateway|"
                           r"received a shutdown signal|lost communication with the server|No space left|Connection reset|"
                           r"TLS handshake timeout|The operation was canceled|Unable to connect|network error"),
           ("FORMAT", r"would reformat|ruff format --check|black --check|would be reformatted|prettier --check|"
                      r"Code style issues found|\[warn\] .* code style|biome format"),
           ("DEPENDENCY", r"ModuleNotFoundError|No matching distribution|ERESOLVE|Cannot find module|ERR_PNPM|"
                          r"frozen-lockfile|lockfile .* (out of date|needs updates)|npm ERR!|ImportError: cannot import"),
           ("SECRET_OR_AUTH", r"Bad credentials|401 Unauthorized|Resource not accessible by integration|"
                              r"secret .* (is not set|not found|missing)|HF_TOKEN|Invalid user token|403 Forbidden"),
           ("PIN_OR_POLICY", r"not pinned to a full-length commit SHA|unpinned action|pin-check|Conventional Commits|"
                             r"commitlint|subject .* does not match|does not match the required pattern"),
           ("DOCTRINE_TEXT", r"overclaim|Doctrine overclaim guard|SZL Doctrine|doctrine_check|forbidden domain"),
           ("LINT", r"ruff check|\bF\d{3}\b|\bE9\d{2}\b|flake8|eslint|oxlint|biome (lint|check)|mypy|pyright|tsc --noEmit|error TS\d+"),
           ("TESTS", r"FAILED tests?/|={3,} FAILURES ={3,}|AssertionError|\b\d+ failed\b|Test Suites?: \d+ failed|"
                     r"Tests:\s+\d+ failed|\u2715|not ok \d+")]
REDACT = re.compile(r"(gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+\S+|token[=:]\s*\S+|"
                    r"https://[^/\s:]+:[^@\s]+@)", re.I)
GATES_DEFAULT = {"leakage": {"max_jaccard5": 0.8, "template_leak_blocks": False, "max_malformed_fraction": 0.02},
                 "schema_validity_min": 0.99, "evidence_handle_validity_min": 1.0, "ece_max": 0.1,
                 "true_abstain_min": 0.9, "false_abstain_max": 0.1, "red_team_refusal_min": 0.95,
                 "quant_parity_min": 0.95, "reproducibility": {"rows": 8},
                 "frozen_note": "Set before any result is seen; sha256 of this file is pinned in candidate.json"}
LANES = []
ACTOR = None


def log(msg):
    line = f"{dt.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    LOGF.write(line + "\n")
    LOGF.flush()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sj(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def wj(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")


def wcsv(p, rows):
    fields = sorted({k for r in rows for k in r}) or ["note"]
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in r.items()})


def mdt(headers, rows):
    def esc(x):
        return str("" if x is None else x).replace("|", "\\|").replace("\n", " ")
    if not rows:
        return "_None._"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def lane(name, fn, *a):
    t = time.time()
    log(f"=== LANE {name} ===")
    try:
        r = fn(*a)
        LANES.append({"lane": name, "status": "OK", "seconds": round(time.time() - t, 1)})
        return r
    except Exception as e:
        (RUN / f"lane_{name}.trace").write_text(traceback.format_exc(), encoding="utf-8")
        LANES.append({"lane": name, "status": "BLOCKED", "error": f"{type(e).__name__}: {e}"[:500],
                      "seconds": round(time.time() - t, 1)})
        log(f"BLOCKED {name}: {type(e).__name__}: {e}")
        return None


class Resp:
    """Minimal stdlib HTTP response (status, text, headers, json, Link-header pagination)."""

    def __init__(self, status, text, headers):
        self.status_code, self.text, self.headers = status, text, headers
        self.ok = 200 <= status < 300

    def json(self):
        return json.loads(self.text)

    @property
    def links(self):
        out = {}
        for part in (self.headers.get("Link") or "").split(","):
            m = re.match(r'\s*<([^>]+)>;\s*rel="([^"]+)"', part)
            if m:
                out[m.group(2)] = {"url": m.group(1)}
        return out


def ok(r):
    return r is not None and r.ok


def http(url, params=None, tok="HF_TOKEN", accept=None, tries=5):
    h = {"User-Agent": "szl-payload/" + VERSION}
    if accept:
        h["Accept"] = accept
    t = os.environ.get(tok) if tok else None
    if t:
        h["Authorization"] = "Bearer " + t
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    r = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90) as resp:
                r = Resp(resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers))
        except urllib.error.HTTPError as e:
            r = Resp(e.code, e.read().decode("utf-8", "replace") if e.fp else "", dict(e.headers or {}))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log(f"WARN network {type(e).__name__} on {url}")
            r = None
            time.sleep(5 * (i + 1))
            continue
        if not (r.status_code in (429, 500, 502, 503, 504) or (r.status_code == 403 and "rate limit" in r.text.lower())):
            return r
        reset = r.headers.get("X-RateLimit-Reset") or r.headers.get("x-ratelimit-reset") or ""
        wait = min(900, max(5, int(reset) - int(time.time())) if reset.isdigit() else 20 * (i + 1))
        log(f"WARN backoff {wait}s on {url}")
        time.sleep(wait)
    return r


def gj(url, params=None, tok="HF_TOKEN"):
    r = http(url, params, tok)
    if not ok(r):
        raise RuntimeError(f"GET {url} -> {getattr(r, 'status_code', 'no response')}")
    return r.json()


def gt(url, tok="HF_TOKEN", accept=None):
    r = http(url, None, tok, accept)
    return r.text[:600000] if ok(r) else ""


def paged(url, params, tok):
    out = []
    while url:
        r = http(url, params, tok)
        if not ok(r):
            raise RuntimeError(f"GET {url} -> {getattr(r, 'status_code', 'no response')}: {getattr(r, 'text', '')[:200]}")
        page = r.json()
        if not isinstance(page, list):
            raise RuntimeError(f"Expected a list from {url}")
        out.extend(page)
        url, params = r.links.get("next", {}).get("url"), None
    return out


def gh(args, inp=None):
    p = subprocess.run([GHBIN] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", input=inp)
    return p.returncode, p.stdout or "", p.stderr or ""


def git(cwd, args, identity=False):
    pre = ["-c", "core.longpaths=true"]
    if identity:
        pre += ["-c", f"user.name={ACTOR or 'szl-estate-payload'}",
                "-c", f"user.email={ACTOR or 'szl-estate-payload'}@users.noreply.github.com"]
    p = subprocess.run([GIT] + pre + (["-C", str(cwd)] if cwd else []) + args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ghapi(path, params=None, paginate=False, raw=False, tries=4):
    """GitHub REST through the authenticated gh CLI (no token handling in this process). Returns parsed JSON,
    raw text when raw=True, or None after logging the failure."""
    args = ["api", path]
    for k, v in (params or {}).items():
        args += ["-X", "GET", "-f", f"{k}={v}"]
    if raw:
        args += ["-H", "Accept: application/vnd.github.raw"]
    if paginate:
        args += ["--paginate", "--slurp"]
    for i in range(tries):
        code, o, e = gh(args)
        if code == 0:
            if raw:
                return o
            try:
                j = json.loads(o or "null")
            except ValueError:
                return None
            if paginate and isinstance(j, list) and j and all(isinstance(x, list) for x in j):
                return [x for page in j for x in page]
            return j
        if re.search(r"rate limit|HTTP 5\d\d|timeout|connection reset", e, re.I):
            log(f"WARN gh api backoff on {path}: {e.strip()[-120:]}")
            time.sleep(20 * (i + 1))
            continue
        log(f"WARN gh api {path} failed: {e.strip()[-200:]}")
        return None
    return None


def forge_raw(path):
    from urllib.parse import quote
    return ghapi(f"repos/{GH_ORG}/{TRAIN_REPO}/contents/{quote(path)}", raw=True) or ""


# ------------------------------------------------------------------ discover
def gh_tree(r):
    if not r.get("default_branch") or not r.get("size"):
        return [], False, None
    j = ghapi(f"repos/{GH_ORG}/{r['name']}/git/trees/{r['default_branch']}?recursive=1")
    if not isinstance(j, dict):
        return [], False, "tree unavailable"
    return [t["path"] for t in j.get("tree", []) if t.get("type") == "blob"], bool(j.get("truncated")), None


def clone(r):
    dest = REPOS / r["name"]
    if dest.exists():
        code, out = git(dest, ["pull", "--ff-only"])
        status = "UPDATED" if code == 0 else f"GIT_FAILED:{code}"
    else:
        code, out = git(None, ["clone", "--depth", "1", r["clone_url"], str(dest)])
        status = "CLONED" if code == 0 else f"GIT_FAILED:{code}"
    if not dest.exists():
        return status, None
    return status, [x.relative_to(dest).as_posix() for x in dest.rglob("*")
                    if x.is_file() and ".git" not in x.relative_to(dest).parts]


def discover():
    global ACTOR
    code, out, _ = gh(["api", "user", "--jq", ".login"])
    ACTOR = out.strip() if code == 0 and out.strip() else None
    est = {"generated_utc": now(), "actor": ACTOR, "models": [], "datasets": [], "spaces": [], "github": [], "forge": {}}
    listed = paged(f"{HF}/api/models", {"author": HF_ORG, "limit": 1000, "full": "true"}, "HF_TOKEN")
    for i, m in enumerate(listed, 1):
        rid = m["id"]
        log(f"[HF model {i}/{len(listed)}] {rid}")
        try:
            d = gj(f"{HF}/api/models/{rid}", {"blobs": "true"})
        except Exception as e:
            d = dict(m, _error=str(e))
        names = [s.get("rfilename", "") for s in d.get("siblings") or []]
        d["_readme"] = gt(f"{HF}/{rid}/raw/main/README.md")
        d["_adapter_cfgs"] = {}
        for n in [n for n in names if bn(n) == "adapter_config.json"][:6]:
            try:
                d["_adapter_cfgs"][n] = json.loads(gt(f"{HF}/{rid}/raw/main/{n}") or "null")
            except ValueError:
                d["_adapter_cfgs"][n] = None
        d["_quant_manifest"] = None
        qm = [n for n in names if bn(n).lower() == "quant_manifest.json"]
        if qm:
            try:
                d["_quant_manifest"] = json.loads(gt(f"{HF}/{rid}/raw/main/{qm[0]}") or "null")
            except ValueError:
                d["_quant_manifest"] = {"_invalid": True}
        est["models"].append(d)
    est["datasets"] = paged(f"{HF}/api/datasets", {"author": HF_ORG, "limit": 1000, "full": "true"}, "HF_TOKEN")
    for s in paged(f"{HF}/api/spaces", {"author": HF_ORG, "limit": 1000, "full": "true"}, "HF_TOKEN"):
        r = http(f"{HF}/api/spaces/{s['id']}/runtime")
        s["_runtime"] = r.json() if ok(r) else {}
        est["spaces"].append(s)
    repos = ghapi(f"orgs/{GH_ORG}/repos", {"type": "all", "per_page": 100}, paginate=True)
    if not isinstance(repos, list):
        raise RuntimeError(f"could not list repositories of {GH_ORG} through gh api")
    for i, r in enumerate(repos, 1):
        log(f"[GitHub {i}/{len(repos)}] {r['name']}")
        paths, trunc, err = gh_tree(r)
        cs = "NOT_REQUESTED"
        if CLONE and not r.get("archived"):
            cs, local = clone(r)
            if local is not None:
                paths, trunc = local, False
        readme = (ghapi(f"repos/{GH_ORG}/{r['name']}/readme", raw=True) or "") if r.get("size") else ""
        est["github"].append({"name": r["name"], "url": r.get("html_url"), "archived": bool(r.get("archived")),
                              "fork": bool(r.get("fork")), "private": bool(r.get("private")),
                              "default_branch": r.get("default_branch"), "language": r.get("language"),
                              "pushed_at": r.get("pushed_at"), "paths": paths, "tree_truncated": trunc,
                              "tree_error": err, "clone_status": cs, "readme": readme})
    est["forge"] = forge_evidence(est)
    wj(RUN / "estate_raw.json", est)
    log(f"OK discovered {len(est['models'])} models, {len(est['datasets'])} datasets, {len(est['spaces'])} spaces, "
        f"{len(est['github'])} GitHub repos; {TRAIN_REPO} evidence: {len(est['forge'].get('paths', []))} paths, "
        f"{len(est['forge'].get('bindings', {}).get('artifacts', []))} source bindings")
    return est


def forge_evidence(est):
    """Authoritative source evidence from the training repo: bindings, portfolio, frontier candidates,
    skip receipts, evaluation identity bindings."""
    repo = next((r for r in est["github"] if r["name"].lower() == TRAIN_REPO.lower()), None)
    ev = {"present": bool(repo), "paths": [], "bindings": {}, "portfolio": {}, "candidates": {}, "skip": [],
          "identity": [], "default_branch": None}
    if not repo:
        return ev
    ev["paths"], ev["default_branch"] = repo["paths"], repo["default_branch"]
    for key, path in (("bindings", "publishing/model-source-bindings.json"), ("portfolio", "portfolio/model_portfolio.json")):
        if path in repo["paths"]:
            try:
                ev[key] = json.loads(forge_raw(path) or "{}")
            except ValueError:
                ev[key] = {}
    for p in repo["paths"]:
        if re.fullmatch(r"frontier/[^/]+/candidate\.json", p):
            try:
                ev["candidates"][p] = json.loads(forge_raw(p) or "{}")
            except ValueError:
                ev["candidates"][p] = {}
        if bn(p) == "skip_receipt.json":
            ev["skip"].append(posixpath.dirname(p))
        m = re.fullmatch(r"frontier/evaluation/identity_bindings/(.+)_[0-9a-f]{12}\.json", p)
        if m:
            ev["identity"].append(m.group(1).lower())
    return ev


# ------------------------------------------------------------------ analyze
def toks(s):
    return {t for t in re.split(r"[-_.\s/]+", s.lower()) if len(t) >= 4 and t not in STOP
            and not re.fullmatch(r"\d+(\.\d+)?b|v\d+|\d+", t)}


def ntoks(s):
    return {t for t in re.split(r"[-_.\s/]+", s.lower()) if t}


def family(n):
    return next((f for f, pats in FAMILIES.items() if any(p in n.lower() for p in pats)), "unmapped")


def norm_base(s):
    s = s.lower().replace("unsloth/", "qwen/")
    for suf in ("-unsloth-bnb-4bit", "-bnb-4bit", "-unsloth"):
        s = s.replace(suf, "")
    return s


def card_score(readme, card):
    heads = "\n".join(line for line in readme.splitlines() if line.lstrip().startswith("#")).lower()
    pats = [r"intended use|uses", r"limitation", r"evaluation|results|benchmark", r"training", r"citation", r"bias|risk|safety"]
    return (sum(1 for p in pats if re.search(p, heads)) + (1 if card.get("license") else 0)) / (len(pats) + 1)


def llm_eligible(d, names):
    """Is this Hub repo a language-model artifact that an SFT kit can target? Conservative: unknown -> False."""
    tags = {str(t).lower() for t in d.get("tags") or []}
    lib = str(d.get("library_name") or "").lower()
    pipe = str(d.get("pipeline_tag") or "").lower()
    if tags & NON_LLM_TAGS or lib in NON_LLM_LIBS:
        return False, "non-llm tags/library: " + ", ".join(sorted((tags & NON_LLM_TAGS) | ({lib} if lib in NON_LLM_LIBS else set())))
    if pipe in LLM_PIPES or lib in LLM_LIBS:
        return True, f"pipeline={pipe or 'n/a'} library={lib or 'n/a'}"
    if any(bn(n) in ("adapter_config.json", "chat_template.jinja", "tokenizer.json", "generation_config.json") for n in names):
        return True, "llm artifact files present"
    return False, "no language-model signal (pipeline/library/adapter/tokenizer)"


def source_map(rid, name, kind, status, ev, llm_ok=True, llm_why=""):
    """Map one Hub model to its training-repo evidence and decide the SFT plan."""
    low = rid.lower()
    portfolio_kind = None
    sm = {"bound_source_path": None, "binding_mode": None, "portfolio_source": None, "folders": [],
          "frontier_candidates": [], "identity_bound": name.lower() in ev.get("identity", []),
          "trainer": [], "evaluator": [], "curriculum": [], "receipts": [], "schema": [], "skip": [], "reference": [],
          "gaps": [], "plan": None}
    for a in (ev.get("bindings") or {}).get("artifacts") or []:
        if str(a.get("repo_id", "")).lower() == low:
            sm["bound_source_path"] = a.get("source_path")
            sm["binding_mode"] = a.get("binding_mode") or "BOUND"
    for a in (ev.get("portfolio") or {}).get("artifacts") or []:
        if str(a.get("repo_id", "")).lower() == low:
            portfolio_kind = a.get("kind")
            if a.get("github_source"):
                sm["portfolio_source"] = a["github_source"]
    sm["portfolio_kind"] = portfolio_kind
    for p, c in (ev.get("candidates") or {}).items():
        ids = {str(c.get("target_repo_id", "")).lower(), str((c.get("predecessor") or {}).get("repo_id", "")).lower()}
        if low in ids:
            sm["frontier_candidates"].append(posixpath.dirname(p))
    paths = ev.get("paths") or []
    dirs = {p.split("/")[0] for p in paths if "/" in p}
    dirs |= {"/".join(p.split("/")[:2]) for p in paths if p.startswith("frontier/") and p.count("/") >= 2}
    mt = ntoks(name)
    mv = {t for t in mt if re.fullmatch(r"[vr]\d+", t)}
    for d in sorted(dirs):
        dn = d.split("/")[-1]
        dtk = ntoks(dn)
        fv = {t for t in dtk if re.fullmatch(r"[vr]\d+", t)}
        sig = {t for t in dtk - GENERIC if not re.fullmatch(r"\d{6,}|c\d", t)}
        if not sig or (fv and (not mv or not (fv & mv))):
            continue
        if re.sub(r"[^a-z0-9]", "", dn.lower()) == re.sub(r"[^a-z0-9]", "", name.lower()) or sig <= mt:
            sm["folders"].append(d)
    for extra in (sm["bound_source_path"], (sm["portfolio_source"] or "").split("/tree/main/", 1)[-1].strip("/")
                  if "/tree/main/" in (sm["portfolio_source"] or "") else None):
        if extra and extra not in sm["folders"]:
            sm["folders"].insert(0, extra)
    scope = sm["folders"] + sm["frontier_candidates"]
    files = [p for p in paths if any(p == f or p.startswith(f + "/") for f in scope)]
    for p in files:
        b = bn(p).lower()
        if re.search(r"(^|/)(train|finetune|sft)[^/]*\.py$|(^|/)launch_supervised_training\.py$", p.lower()):
            sm["trainer"].append(p)
        if re.search(r"(^|/)(eval|evaluate)[^/]*\.py$", p.lower()):
            sm["evaluator"].append(p)
        if b.endswith(".jsonl"):
            sm["curriculum"].append(p)
        if re.search(r"receipt[^/]*\.json$", b):
            sm["receipts"].append(p)
        if b.endswith(".schema.json"):
            sm["schema"].append(p)
        if b == "skip_receipt.json":
            sm["skip"].append(p)
        if b == "load.py":
            sm["reference"].append(p)
    if kind == "SOFTWARE" or portfolio_kind == "software_kernel":
        sm["plan"] = "SOFTWARE_NO_SFT"
    elif name.lower() == HF_ORG.lower():
        sm["plan"] = "ORG_PROFILE_NOT_A_MODEL"
    elif status == "QUANT_ONLY":
        sm["plan"] = "QUANT_DERIVATIVE_PARITY_ONLY"
    elif sm["skip"]:
        sm["plan"] = "SKIPPED_BY_RECEIPT"
    elif sm["trainer"]:
        sm["plan"] = "HAS_TRAINER"
        sm["gaps"] = [g for g, present in (("evaluator", sm["evaluator"]), ("curriculum", sm["curriculum"]),
                                           ("receipts", sm["receipts"]), ("schema", sm["schema"])) if not present]
    elif sm["frontier_candidates"]:
        sm["plan"] = "FRONTIER_CANDIDATE_EXISTS"
    elif sm["reference"]:
        sm["plan"] = "REFERENCE_ARTIFACT_NO_SFT"
    elif not llm_ok:
        sm["plan"] = "NON_LLM_OR_UNDEFINED_ARTIFACT"
        sm["gaps"] = [llm_why]
    else:
        sm["plan"] = "GENERATE_CANDIDATE"
    return sm


def analyze(est):
    ev = est.get("forge") or {}
    all_ids = {x["id"].lower() for k in ("models", "datasets", "spaces") for x in est[k]}
    model_mod = {m["id"]: m.get("lastModified") or "" for m in est["models"]}
    gh_repos = {r["name"].lower(): r for r in est["github"]}
    gpaths = {k: [p.lower() for p in r["paths"]] for k, r in gh_repos.items()}
    ghlinks = {k: {x.lower().rstrip(".") for x in HF_LINK.findall(r.get("readme") or "")} for k, r in gh_repos.items()}
    gtoks = {k: toks(k) for k in gh_repos}
    models, drift, sync = [], [], []
    for d in est["models"]:
        rid = d["id"]
        name = rid.split("/")[-1]
        size = {s["rfilename"]: int(s.get("size") or (s.get("lfs") or {}).get("size") or 0)
                for s in (d.get("siblings") or []) if s.get("rfilename")}
        names = list(size)
        card = d.get("cardData") if isinstance(d.get("cardData"), dict) else {}
        readme = d.get("_readme") or ""
        acfg = [n for n in names if bn(n) == "adapter_config.json"]
        aw = [n for n in names if bn(n).startswith("adapter_model")]
        gg = [n for n in names if n.lower().endswith(".gguf")]
        fw = [n for n in names if n.lower().endswith(WEIGHT_EXT) and not bn(n).startswith("adapter_model") and bn(n) not in NON_WEIGHT]
        tensorish = [n for n in fw + aw + gg if n.lower().endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth"))]
        tiny = [n for n in tensorish if 0 < size.get(n, 0) < 1_000_000]
        orphan = sorted({posixpath.dirname(n) or "." for n in acfg} - {posixpath.dirname(n) or "." for n in aw})
        code = [n for n in names if n.lower().endswith((".py", ".rs", ".cpp", ".cu", ".lean", ".ts"))]
        rec = [n for n in names if RECEIPT_RE.search(bn(n))]
        if fw:
            status = "WEIGHTED"
        elif aw:
            status = "ADAPTER"
        elif gg:
            status = "QUANT_ONLY"
        elif acfg or any(bn(n) == "config.json" for n in names):
            status = "MISSING_WEIGHTS"
        elif any(h in name.lower() for h in LIB_HINTS) or len(code) >= 2:
            status = "SOFTWARE"
        elif names:
            status = "CARD_ONLY"
        else:
            status = "EMPTY"
        kind = "SOFTWARE" if status == "SOFTWARE" else "MODEL"
        flags = []
        if tensorish and len(tiny) == len(tensorish):
            flags.append("PLACEHOLDER_WEIGHTS")
        if orphan:
            flags.append("ORPHAN_ADAPTER_CONFIG:" + ",".join(orphan))
        if gg and (acfg or aw) and not fw:
            flags.append("MIXED_ADAPTER_AND_GGUF")
        bm = card.get("base_model")
        bases = [bm] if isinstance(bm, str) else [str(x) for x in (bm or [])]
        abases = sorted({str(c.get("base_model_name_or_path")) for c in (d.get("_adapter_cfgs") or {}).values()
                         if isinstance(c, dict) and c.get("base_model_name_or_path")})
        mismatch = bool(bases and abases and not ({norm_base(x) for x in bases} & {norm_base(x) for x in abases}))
        if mismatch:
            flags.append("LINEAGE_MISMATCH")
        if kind == "MODEL" and not bases:
            flags.append("NO_BASE_MODEL_DECLARED")
        if not card.get("license"):
            flags.append("NO_LICENSE")
        if kind == "MODEL" and not rec:
            flags.append("NO_RECEIPTS")
        linked = {re.sub(r"\.git$", "", x.lower()).rstrip(".") for x in GH_LINK.findall(readme)}
        backlinked = {k for k, v in ghlinks.items() if rid.lower() in v}
        tokmatch = {k for k, v in gtoks.items() if toks(name) & v}
        sources = sorted((linked | backlinked | tokmatch) & set(gh_repos))
        for k in sorted(linked - set(gh_repos)):
            sync.append({"asset": rid, "issue": "BROKEN_SOURCE_LINK", "detail": f"{GH_ORG}/{k}"})
        llm_ok, llm_why = llm_eligible(d, names)
        sm = source_map(rid, name, kind, status, ev, llm_ok, llm_why)
        if kind == "MODEL" and not (linked & set(gh_repos)) and not backlinked and not sm["folders"]:
            sync.append({"asset": rid, "issue": "NO_SOURCE_LINK", "detail": "card links no org repo, no repo links back, no training-repo folder"})
        dflags = []
        if gg:
            levels = sorted({m.group(1).upper() for n in gg for m in [QRE.search(bn(n))] if m})
            parents = [b for b in bases if b.lower().startswith(HF_ORG.lower() + "/")]
            if not parents and not fw and not aw:
                dflags.append("UNLINKED_QUANT")
            for p in parents:
                pm = next((i for i in model_mod if i.lower() == p.lower()), None)
                if not pm:
                    dflags.append("PARENT_NOT_FOUND:" + p)
                elif model_mod[pm] > (d.get("lastModified") or ""):
                    dflags.append("PARENT_NEWER_THAN_QUANT:" + pm)
            if not {"F16", "BF16", "F32"} & set(levels) and not fw:
                dflags.append("NO_FULL_PRECISION_REFERENCE")
            qm = d.get("_quant_manifest")
            if not isinstance(qm, dict) or not qm.get("parent_revision"):
                dflags.append("UNPINNED_PARENT_REVISION")
            if not any("parity" in bn(n).lower() for n in rec):
                dflags.append("NO_PARITY_RECEIPT")
            drift.append({"model": rid, "levels": ",".join(levels), "parents": ",".join(parents),
                          "mmproj": any("mmproj" in n.lower() for n in gg), "flags": dflags})
        src_pts = 20 if sm["plan"] == "HAS_TRAINER" and not sm["gaps"] else 12 if sm["plan"] in ("HAS_TRAINER", "FRONTIER_CANDIDATE_EXISTS") else 0
        if kind == "MODEL":
            w = 0 if "PLACEHOLDER_WEIGHTS" in flags else {"WEIGHTED": 30, "ADAPTER": 30, "QUANT_ONLY": 15}.get(status, 0)
            score = round(w + 15 * card_score(readme, card) + (5 if card.get("license") else 0)
                          + (10 if bases and not mismatch else 5 if bases else 0) + src_pts
                          + (10 if rec else 0) + (max(0.0, 10 - 2.5 * len(dflags)) if gg else 10), 1)
            tier = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D"
        else:
            score, tier = None, "SOFTWARE"
        nxt = {"GENERATE_CANDIDATE": "CANDIDATE_KIT_GENERATED_BIND_DATA_THEN_TRAIN",
               "HAS_TRAINER": "RUN_EXISTING_TRAINER_PATH_CLOSE_GAPS:" + ",".join(sm["gaps"]) if sm["gaps"] else "VERIFY_RELEASE_GATES_WITH_EXISTING_TRAINER",
               "FRONTIER_CANDIDATE_EXISTS": "RUN_EXISTING_FRONTIER_CANDIDATE",
               "SKIPPED_BY_RECEIPT": "HONESTLY_SKIPPED_KEEP_RECEIPT",
               "REFERENCE_ARTIFACT_NO_SFT": "RUN_LOAD_CONTRACT_NOT_SFT",
               "QUANT_DERIVATIVE_PARITY_ONLY": "TRACE_PARENT_AND_PARITY_RECEIPT",
               "SOFTWARE_NO_SFT": "RELABEL_AS_SOFTWARE_AND_TEST",
               "ORG_PROFILE_NOT_A_MODEL": "KEEP_AS_ORG_STUB",
               "NON_LLM_OR_UNDEFINED_ARTIFACT": "DEFINE_ARTIFACT_CONTRACT_OR_RETIRE"}[sm["plan"]]
        if status in ("MISSING_WEIGHTS", "CARD_ONLY", "EMPTY") or "PLACEHOLDER_WEIGHTS" in flags:
            nxt = "QUARANTINE_THEN_" + nxt
        models.append({"model": rid, "family": family(name), "kind": kind, "status": status, "score": score,
                       "tier": tier, "next_action": nxt, "flags": flags, "drift_flags": dflags, "sources": sources,
                       "base_model": bases, "adapter_base": abases, "license": card.get("license"), "receipts": rec,
                       "weights": len(fw), "adapters": len(aw), "gguf": len(gg), "downloads": d.get("downloads", 0),
                       "last_modified": d.get("lastModified"), "sha": d.get("sha"), "schema_files": [n for n in names if n.endswith(".schema.json")],
                       "sft_plan": sm["plan"], "forge_folders": sm["folders"], "forge_trainer": sm["trainer"][:4],
                       "forge_evaluator": sm["evaluator"][:4], "forge_receipts": len(sm["receipts"]),
                       "forge_curriculum": len(sm["curriculum"]), "forge_gaps": sm["gaps"],
                       "forge_binding": sm["bound_source_path"], "frontier_candidates": sm["frontier_candidates"],
                       "identity_bound": sm["identity_bound"], "portfolio_kind": sm.get("portfolio_kind"),
                       "pipeline_tag": d.get("pipeline_tag"), "library_name": d.get("library_name"), "llm_eligible": llm_ok})
    datasets = []
    for x in est["datasets"]:
        card = x.get("cardData") if isinstance(x.get("cardData"), dict) else {}
        tags = [str(t).lower() for t in x.get("tags") or []]
        syn = any(h in x["id"].lower() + " " + " ".join(tags) for h in SYN_HINTS)
        lic = card.get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
        datasets.append({"dataset": x["id"], "license": lic, "synthetic_signal": syn,
                         "status": "EXCLUDE_SYNTHETIC_OR_FIXTURE" if syn else "EXCLUDE_LICENSE_UNKNOWN" if not lic
                         else "CANDIDATE_NEEDS_PROVENANCE_AND_LEAKAGE_GATE"})
    spaces = []
    for s in est["spaces"]:
        stage = (s.get("_runtime") or {}).get("stage", "UNKNOWN")
        spaces.append({"space": s["id"], "sdk": s.get("sdk"), "stage": stage,
                       "broken": stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE")})
    repos = []
    for k, r in gh_repos.items():
        ps = gpaths[k]
        fn = [bn(p) for p in ps]
        wf = sum(p.startswith(".github/workflows/") and p.endswith((".yml", ".yaml")) for p in ps)
        tests = sum(bool(re.search(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+\.py$|\.(test|spec)\.[jt]sx?$", p)) for p in ps)
        gaps = [n for n, present in (("readme", any(x.startswith("readme") for x in fn)),
                                     ("license", any(x.startswith(("license", "licence", "copying")) for x in fn)),
                                     ("security", "security.md" in fn), ("codeowners", "codeowners" in fn),
                                     ("ci", wf > 0), ("tests", tests > 0)) if not present]
        for x in sorted(ghlinks[k] - all_ids):
            sync.append({"asset": f"{GH_ORG}/{r['name']}", "issue": "BROKEN_ARTIFACT_LINK", "detail": x})
        repos.append({"repo": f"{GH_ORG}/{r['name']}", "archived": r["archived"], "fork": r["fork"],
                      "private": r["private"], "language": r["language"], "pushed_at": r["pushed_at"],
                      "files": len(ps), "tree_truncated": r["tree_truncated"], "clone_status": r["clone_status"],
                      "workflows": wf, "tests": tests, "governance_gaps": [] if r["archived"] else gaps})
    log("source plans: " + str(dict(Counter(m["sft_plan"] for m in models))))
    return {"models": models, "datasets": datasets, "spaces": spaces, "repos": repos, "drift": drift, "sync": sync}


# ------------------------------------------------------------------ ledger (quarantine + repair idempotency)
def ledger_entries():
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] if LEDGER.exists() else []


def ledger_verify():
    prev, n = "0" * 64, 0
    for e in ledger_entries():
        n += 1
        claimed = e.pop("entry_sha256", None)
        if e.get("prev_sha256") != prev or sha(sj(e)) != claimed:
            return {"valid": False, "broken_at_entry": n}
        prev = claimed
    return {"valid": True, "entries": n, "head": prev}


def ledger_append(ev):
    es = ledger_entries()
    e = {"ts": now(), "prev_sha256": es[-1]["entry_sha256"] if es else "0" * 64, **ev}
    e["entry_sha256"] = sha(sj(e))
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(sj(e) + "\n")


def ledger_has(fp):
    return any(e.get("fingerprint") == fp for e in ledger_entries())


# ------------------------------------------------------------------ quarantine
def banner(readme, reason):
    if "QUARANTINED (" in readme:
        return None
    note = (f"> **QUARANTINED ({reason})** - automated SZL estate audit, {now()}. {MSG.get(reason, reason)} "
            "Do not use in production until real artifacts are published and a passing release-gate receipt "
            "is attached. See `QUARANTINE.json`.\n\n")
    if readme.startswith("---"):
        end = readme.find("\n---", 3)
        if end != -1:
            nl = readme.find("\n", end + 4)
            cut = len(readme) if nl == -1 else nl + 1
            return readme[:cut] + "\n" + note + readme[cut:]
    return note + readme


def open_hf_pr(kind, rid, reason):
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    title = f"[SZL quarantine] {reason}"
    try:
        for disc in api.get_repo_discussions(repo_id=rid, repo_type=kind):
            if disc.is_pull_request and disc.status == "open" and disc.title == title:
                return "PR_ALREADY_OPEN", disc.num
    except Exception:
        pass
    readme = gt(f"{HF}/{'datasets/' if kind == 'dataset' else ''}{rid}/raw/main/README.md")
    record = {"schema": "szl.quarantine/v1", "repo_id": rid, "repo_type": kind, "reason": reason,
              "detail": MSG.get(reason, reason), "created_utc": now(), "payload_version": VERSION,
              "release_condition": "Publish real artifacts, attach a passing twelve-gate receipt, then lift in a reviewed PR."}
    ops = [CommitOperationAdd(path_in_repo="QUARANTINE.json", path_or_fileobj=json.dumps(record, indent=2).encode("utf-8"))]
    new = banner(readme, reason)
    if new is not None:
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=new.encode("utf-8")))
    info = api.create_commit(repo_id=rid, repo_type=kind, operations=ops, commit_message=title,
                             commit_description=MSG.get(reason, reason), create_pr=True)
    return "PR_OPENED", getattr(info, "pr_url", None)


def quarantine(A):
    v = ledger_verify()
    if not v["valid"]:
        raise RuntimeError(f"Ledger hash chain broken: {v}. Refusing to extend it.")
    hist = ledger_entries()
    seen = {e.get("fingerprint") for e in hist}
    opened = {e.get("fingerprint") for e in hist if e.get("action") in ("PR_OPENED", "PR_ALREADY_OPEN")}
    held = {e.get("repo_id") for e in hist if e.get("action") in ("PLANNED", "PR_OPENED", "PR_ALREADY_OPEN")}
    items = []
    for m in A["models"]:
        reason = "PLACEHOLDER_WEIGHTS" if "PLACEHOLDER_WEIGHTS" in m["flags"] else (
            m["status"] if m["status"] in ("MISSING_WEIGHTS", "CARD_ONLY", "EMPTY") else None)
        if reason:
            items.append(("model", m["model"], reason))
        elif m["model"] in held and m["status"] in ("WEIGHTED", "ADAPTER"):
            items.append(("model", m["model"], "RELEASE_ELIGIBLE"))
    items += [("dataset", x["dataset"], x["status"].replace("EXCLUDE_", "")) for x in A["datasets"] if x["status"].startswith("EXCLUDE_")]
    out = []
    for kind, rid, reason in items:
        fp = sha(f"{kind}|{rid}|{reason}")
        row = {"repo_type": kind, "repo_id": rid, "reason": reason}
        if fp not in seen:
            ledger_append({"action": "RELEASE_ELIGIBLE" if reason == "RELEASE_ELIGIBLE" else "PLANNED",
                           "repo_id": rid, "repo_type": kind, "reason": reason, "fingerprint": fp})
        if reason == "RELEASE_ELIGIBLE":
            out.append({**row, "action": "LIFT_ONLY_WITH_PASSING_GATE_RECEIPT"})
        elif fp in opened:
            out.append({**row, "action": "PR_ALREADY_RECORDED"})
        elif not EXECUTE:
            out.append({**row, "action": "DRY_RUN"})
        else:
            try:
                act, ref = open_hf_pr(kind, rid, reason)
                ledger_append({"action": act, "repo_id": rid, "repo_type": kind, "reason": reason, "fingerprint": fp, "ref": str(ref)})
                out.append({**row, "action": act, "ref": ref})
                log(f"quarantine {act}: {rid} ({reason}) {ref}")
            except Exception as e:
                out.append({**row, "action": "PR_FAILED", "ref": f"{type(e).__name__}: {e}"[:300]})
    log(f"ledger: {ledger_verify()}")
    return out


# ------------------------------------------------------------------ pull requests (green-only)
def checks(roll):
    t = pa = fa = pe = 0
    for x in roll or []:
        if not isinstance(x, dict):
            continue
        t += 1
        if x.get("__typename") == "StatusContext" or ("state" in x and "conclusion" not in x):
            s = str(x.get("state") or "").upper()
            pa, fa, pe = (pa + 1, fa, pe) if s == "SUCCESS" else (pa, fa + 1, pe) if s in ("FAILURE", "ERROR") else (pa, fa, pe + 1)
        else:
            st, c = str(x.get("status") or "").upper(), str(x.get("conclusion") or "").upper()
            if st and st != "COMPLETED":
                pe += 1
            elif c in PASS:
                pa += 1
            elif c in FAIL:
                fa += 1
            else:
                pe += 1
    return {"total": t, "passed": pa, "failed": fa, "pending": pe}


def is_hold(title, labels):
    return bool(re.search(r"(^|[\s\[(])hold([\s\]):]|$)", title or "", re.I) or re.search(r"(^|;)hold(;|$)", labels, re.I))


def green(pr):
    labels = ";".join(lb.get("name", "") if isinstance(lb, dict) else str(lb) for lb in pr.get("labels") or [])
    c = checks(pr.get("statusCheckRollup"))
    return (not pr.get("isDraft") and not is_hold(pr.get("title"), labels) and pr.get("mergeStateStatus") in ("CLEAN", "HAS_HOOKS")
            and c["total"] > 0 and c["failed"] == 0 and c["pending"] == 0), c, labels


def handle_pr(full, pr):
    n = str(pr["number"])
    _, c, labels = green(pr)
    ms = pr.get("mergeStateStatus")
    row = {"repo": full, "number": pr["number"], "title": pr.get("title"), "url": pr.get("url"), "head_sha": pr.get("headRefOid"),
           "merge_state": ms, "labels": labels, **{f"checks_{k}": v for k, v in c.items()}}
    if pr.get("isDraft"):
        return {**row, "result": "DRAFT_PRESERVED"}
    if is_hold(pr.get("title"), labels):
        return {**row, "result": "HOLD_PRESERVED"}
    if ms == "BEHIND":
        if not MERGE:
            return {**row, "result": "BEHIND_DRY_RUN"}
        code, o, e = gh(["pr", "update-branch", n, "--repo", full])
        return {**row, "result": "BRANCH_UPDATED_RECHECK_NEXT_RUN" if code == 0 else "UPDATE_BRANCH_FAILED", "detail": (o + e)[:300]}
    if ms == "DIRTY":
        return {**row, "result": "CONFLICT_NEEDS_REBASE"}
    if ms in ("BLOCKED", "UNSTABLE"):
        if c["failed"]:
            return {**row, "result": "FAILING_CHECKS"}
        if c["pending"]:
            return {**row, "result": "PENDING_CHECKS"}
        return {**row, "result": f"NOT_MERGEABLE_{ms}", "detail": "all checks green; a protection rule (required review, "
                                                                   "required check context, or signature) blocks the merge"}
    if ms not in ("CLEAN", "HAS_HOOKS"):
        return {**row, "result": f"NOT_MERGEABLE_{ms}"}
    if c["total"] == 0:
        return {**row, "result": "NO_CHECKS_NOT_MERGED"}
    if c["failed"]:
        return {**row, "result": "FAILING_CHECKS"}
    if c["pending"]:
        return {**row, "result": "PENDING_CHECKS"}
    if not MERGE:
        return {**row, "result": "ELIGIBLE_DRY_RUN"}
    code, o, e = gh(["pr", "view", n, "--repo", full, "--json", "number,title,isDraft,mergeStateStatus,headRefOid,statusCheckRollup,labels"])
    if code:
        return {**row, "result": "LIVE_RECHECK_FAILED", "detail": e[:300]}
    live = json.loads(o)
    if not green(live)[0] or live.get("headRefOid") != pr.get("headRefOid"):
        return {**row, "result": "CHANGED_DURING_RECHECK_SKIPPED"}
    last = ""
    for method in ("--squash", "--merge", "--rebase"):
        code, o, e = gh(["pr", "merge", n, "--repo", full, method, "--delete-branch", "--match-head-commit", live["headRefOid"]])
        last = (o + e)[:300]
        if code == 0:
            log(f"MERGED {full}#{n} ({method})")
            return {**row, "result": "MERGED", "method": method}
        if "not allowed" not in last.lower():
            break
    return {**row, "result": "MERGE_FAILED", "detail": last}


def pr_lane(est):
    rows = []
    for r in est["github"]:
        if r["archived"]:
            continue
        full = f"{GH_ORG}/{r['name']}"
        code, o, e = gh(["pr", "list", "--repo", full, "--state", "open", "--limit", "200", "--json",
                         "number,title,url,isDraft,mergeStateStatus,headRefOid,labels,statusCheckRollup"])
        if code:
            rows.append({"repo": full, "result": "LIST_FAILED", "detail": e[:300]})
            continue
        for pr in json.loads(o or "[]"):
            try:
                rows.append(handle_pr(full, pr))
            except Exception as ex:
                rows.append({"repo": full, "number": pr.get("number"), "result": "ERROR", "detail": str(ex)[:300]})
        time.sleep(0.2)
    log(f"PR results: {dict(Counter(x['result'] for x in rows))}")
    return rows


# ------------------------------------------------------------------ pr_repair
def classify(text):
    hits = [name for name, rx in CLASSES if re.search(rx, text, re.I | re.M)]
    return hits or ["UNCLASSIFIED"]


def redact(text):
    return REDACT.sub("[REDACTED]", text)


def excerpt(text, hits=60, tail=50, limit=12000):
    """Failure-focused log excerpt: lines around error markers plus the tail, timestamps stripped."""
    lines = [re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z ?", "", ln) for ln in text.splitlines()]
    marker = re.compile(r"##\[error\]|\berror\b|\bfail(ed|ure|ing)?\b|\u2716|\u2715|exception|traceback|assert|not ok", re.I)
    keep = set()
    for i in [i for i, ln in enumerate(lines) if marker.search(ln)][:hits]:
        keep.update(range(max(0, i - 2), min(len(lines), i + 3)))
    keep.update(range(max(0, len(lines) - tail), len(lines)))
    out, prev = [], -1
    for i in sorted(keep):
        if prev != -1 and i != prev + 1:
            out.append("...")
        out.append(lines[i])
        prev = i
    return "\n".join(out)[-limit:]


def repo_checkout(full):
    dest = REPOS / full.split("/")[1]
    if (dest / ".git").exists():
        code, out = git(dest, ["fetch", "origin", "--prune"])
    else:
        code, out = git(None, ["clone", "--filter=blob:none", f"https://github.com/{full}.git", str(dest)])
    if code != 0:
        raise RuntimeError(f"git checkout of {full} failed: {out[-400:]}")
    return dest


def checkout_pr_head(dest, head_ref, head_sha):
    code, out = git(dest, ["fetch", "origin", head_ref])
    if code != 0:
        raise RuntimeError(f"fetch {head_ref}: {out[-300:]}")
    code, out = git(dest, ["checkout", "-B", head_ref, "FETCH_HEAD"])
    if code != 0:
        raise RuntimeError(f"checkout {head_ref}: {out[-300:]}")
    code, out = git(dest, ["rev-parse", "HEAD"])
    if out.strip() != head_sha:
        raise RuntimeError(f"head moved: expected {head_sha}, got {out.strip()}")


def repair_once(fp, action):
    """Idempotency: one action per (repo, PR, head SHA, action kind)."""
    if ledger_has(fp):
        return False
    ledger_append({**action, "fingerprint": fp})
    return True


def comment_pr(full, n, head_sha, body):
    fp = sha(f"comment|{full}|{n}|{head_sha}")
    if not COMMENT or not REPAIR:
        return "COMMENT_DRY_RUN"
    if ledger_has(fp):
        return "COMMENT_ALREADY_POSTED"
    code, o, e = gh(["pr", "comment", str(n), "--repo", full, "--body", body])
    if code == 0:
        ledger_append({"action": "PR_COMMENT", "repo": full, "pr": n, "head_sha": head_sha, "fingerprint": fp})
        return "COMMENT_POSTED"
    return "COMMENT_FAILED:" + (o + e)[-200:]


def formatter_for(dest):
    py = (dest / "pyproject.toml")
    txt = py.read_text(encoding="utf-8", errors="replace") if py.is_file() else ""
    if (dest / "ruff.toml").is_file() or (dest / ".ruff.toml").is_file() or "[tool.ruff" in txt:
        return ["ruff", "format", "."]
    if "[tool.black]" in txt:
        return ["black", "."]
    return None


def write_brief(full, pr, classes, logs, note):
    n = pr["number"]
    files = [f.get("path") for f in pr.get("files") or []][:60]
    name = f"{full.replace('/', '__')}__{n}.md"
    p = RUN / "repairs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# Repair brief: {full}#{n} - {pr.get('title')}", "", f"URL: {pr.get('url')}", f"Head: `{pr.get('headRefOid')}` "
            f"({pr.get('headRefName')} -> {pr.get('baseRefName')}) | author {((pr.get('author') or {}).get('login'))} | "
            f"merge state {pr.get('mergeStateStatus')} | cross-repository {pr.get('isCrossRepository')}", "",
            f"Classification: **{', '.join(classes)}**", "", f"Payload action: {note}", "",
            "## Files changed by the PR", "", "\n".join(f"- `{f}`" for f in files) or "_unknown_", "",
            "## Failing jobs (redacted log tails)", ""]
    for job, tail in logs:
        body += [f"### {job}", "", "```text", redact(tail), "```", ""]
    body += ["## Instructions for the coding agent", "",
             "1. Reproduce the failing job locally from the PR head (`gh pr checkout`).",
             "2. Fix the root cause named above; do not weaken, skip, or delete the failing check.",
             "3. If the failing gate is a doctrine or proof gate, treat red as a finding, not a bug in the gate.",
             "4. Push small commits; the payload merges only when every check is complete and green on a live re-check.", ""]
    p.write_text("\n".join(body), encoding="utf-8")
    return str(p)


def repair_pr(full, number, prior_detail=""):
    code, o, e = gh(["pr", "view", str(number), "--repo", full, "--json",
                     "number,title,url,isDraft,headRefName,headRefOid,baseRefName,isCrossRepository,author,labels,"
                     "mergeStateStatus,statusCheckRollup,files"])
    if code:
        return {"repo": full, "number": number, "action": "VIEW_FAILED", "detail": e[:200]}
    pr = json.loads(o)
    head = pr.get("headRefOid")
    author = str((pr.get("author") or {}).get("login") or "")
    dependabot = "dependabot" in author.lower() or str(pr.get("headRefName", "")).startswith("dependabot/")
    row = {"repo": full, "number": number, "title": pr.get("title"), "url": pr.get("url"), "head_sha": head,
           "merge_state": pr.get("mergeStateStatus"), "dependabot": dependabot, "cross_repo": bool(pr.get("isCrossRepository"))}
    failed = [x for x in pr.get("statusCheckRollup") or [] if isinstance(x, dict)
              and (str(x.get("conclusion") or "").upper() in FAIL or str(x.get("state") or "").upper() in ("FAILURE", "ERROR"))]
    run_ids, logs = [], []
    for x in failed:
        m = re.search(r"/actions/runs/(\d+)", str(x.get("detailsUrl") or x.get("targetUrl") or ""))
        if m and m.group(1) not in run_ids:
            run_ids.append(m.group(1))
    for rid in run_ids[:5]:
        code, o, e = gh(["run", "view", rid, "--repo", full, "--log-failed"])
        logs.append((f"run {rid}", excerpt(o + e)))
    joined = "\n".join(t for _, t in logs) or "\n".join(str(x.get("name")) + " " + str(x.get("description") or "") for x in failed)
    classes = classify(joined) if (failed or logs) else []
    conflict = pr.get("mergeStateStatus") == "DIRTY"
    row["classes"] = classes
    row["failed_checks"] = [str(x.get("name") or x.get("context")) for x in failed][:12]
    dry = not REPAIR

    def do(kind, fn):
        fp = sha(f"repair|{full}|{number}|{head}|{kind}")
        if dry:
            return kind + "_DRY_RUN"
        if ledger_has(fp):
            return kind + "_ALREADY_DONE_FOR_THIS_HEAD"
        result = fn()
        ledger_append({"action": kind, "repo": full, "pr": number, "head_sha": head, "result": result, "fingerprint": fp})
        return result

    try:
        if classes and set(classes) <= {"INTENTIONAL_RED"}:
            row["action"] = "INTENTIONAL_RED_PRESERVED"
            row["detail"] = "designed honesty gate (proof bounty) stays red; nothing to fix"
            return row
        if conflict and dependabot:
            row["action"] = do("DEPENDABOT_REBASE", lambda: comment_pr(full, number, head, "@dependabot rebase"))
            return row
        if conflict and not pr.get("isCrossRepository"):
            def merge_base():
                dest = repo_checkout(full)
                checkout_pr_head(dest, pr["headRefName"], head)
                code, out = git(dest, ["merge", "--no-edit", f"origin/{pr['baseRefName']}"], identity=True)
                if code == 0:
                    code, out = git(dest, ["push", "origin", f"HEAD:{pr['headRefName']}"])
                    return "BASE_MERGED_AND_PUSHED_RECHECK_NEXT_RUN" if code == 0 else "PUSH_FAILED:" + out[-200:]
                _, files_out = git(dest, ["diff", "--name-only", "--diff-filter=U"])
                git(dest, ["merge", "--abort"])
                conflicts = [f for f in files_out.splitlines() if f.strip()]
                row["conflicting_files"] = conflicts
                brief = write_brief(full, pr, ["MERGE_CONFLICT"], logs, f"base merge conflicts in {len(conflicts)} files: {conflicts[:10]}")
                row["brief"] = brief
                comment_pr(full, number, head, f"SZL estate payload: this branch conflicts with `{pr['baseRefName']}` in "
                           f"{len(conflicts)} file(s): " + ", ".join(f"`{c}`" for c in conflicts[:10]) + ". A rebase with manual resolution is needed; "
                           "the payload does not resolve conflicts by taking either side.")
                return "CONFLICT_UNRESOLVED_BRIEF_WRITTEN"
            row["action"] = do("MERGE_BASE", merge_base)
            return row
        if conflict:
            row["action"] = "CONFLICT_CROSS_REPO_BRIEF"
            row["brief"] = write_brief(full, pr, ["MERGE_CONFLICT_FORK"], logs, "fork PR conflicts; the fork owner must rebase")
            return row
        if not failed:
            if prior_detail:
                row["brief"] = write_brief(full, pr, ["REVIEW_OR_PROTECTION_REQUIRED"], [("merge attempt", prior_detail)],
                                           "GitHub blocks the merge without a failing check: a required review, required check "
                                           "context, or signature rule applies. Owner action, not a code fix.")
                row["action"] = "REVIEW_OR_PROTECTION_REQUIRED_BRIEF"
            else:
                row["action"] = "NO_FAILED_CHECKS_NOW_RECHECK_NEXT_RUN"
            return row
        if set(classes) <= {"INFRA_FLAKE"} and run_ids:
            def rerun():
                res = []
                for rid in run_ids[:5]:
                    code, o, e = gh(["run", "rerun", rid, "--repo", full, "--failed"])
                    res.append(f"{rid}:{'ok' if code == 0 else (o + e)[-80:]}")
                return "RERUN_TRIGGERED " + " ".join(res)
            row["action"] = do("RERUN_FLAKE", rerun)
            return row
        if dependabot:
            row["action"] = do("DEPENDABOT_RECREATE", lambda: comment_pr(full, number, head, "@dependabot recreate"))
            return row
        if "FORMAT" in classes and not pr.get("isCrossRepository"):
            def fmt():
                dest = repo_checkout(full)
                checkout_pr_head(dest, pr["headRefName"], head)
                cmd = formatter_for(dest)
                if not cmd:
                    return "FORMAT_TOOL_UNKNOWN_BRIEF"
                exe = shutil.which(cmd[0]) or str(pathlib.Path(sys.executable).parent / cmd[0])
                p = subprocess.run([exe] + cmd[1:], cwd=str(dest), capture_output=True, text=True)
                if p.returncode != 0:
                    return "FORMATTER_FAILED:" + (p.stdout + p.stderr)[-200:]
                code, out = git(dest, ["status", "--porcelain"])
                if not out.strip():
                    return "FORMAT_NO_CHANGE"
                git(dest, ["add", "-A"])
                code, out = git(dest, ["commit", "-m", "style: apply repository formatter (automated by SZL estate payload)"], identity=True)
                if code != 0:
                    return "COMMIT_FAILED:" + out[-200:]
                code, out = git(dest, ["push", "origin", f"HEAD:{pr['headRefName']}"])
                return "FORMAT_FIX_PUSHED" if code == 0 else "PUSH_FAILED:" + out[-200:]
            row["action"] = do("FORMAT_FIX", fmt)
            if row["action"].startswith("FORMAT_FIX_PUSHED"):
                return row
            row["detail"] = row["action"]
        note = "needs code judgment; brief written for the coding agent"
        row["brief"] = write_brief(full, pr, classes, logs, note)
        row["action"] = "BRIEF_WRITTEN"
        row["comment"] = comment_pr(full, number, head, f"SZL estate payload: failing checks classified as **{', '.join(classes)}** "
                                    f"({', '.join(row['failed_checks'][:6])}). A repair brief with redacted log tails was written for the "
                                    "coding agent; the payload does not weaken or skip gates.")
        return row
    except Exception as ex:
        row["action"] = "REPAIR_ERROR"
        row["detail"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        return row


def pr_repair(P):
    targets = [p for p in P if p.get("result") in REPAIR_RESULTS and p.get("number")]
    out = []
    for p in targets:
        log(f"repair {p['repo']}#{p['number']} ({p['result']})")
        out.append({**repair_pr(p["repo"], p["number"], str(p.get("detail") or "")), "pr_result": p["result"]})
    log(f"repair actions: {dict(Counter(str(x.get('action'))[:40] for x in out))}")
    return out


# ------------------------------------------------------------------ corpora (local owner data)
def scan_corpora():
    if kit is None:
        raise RuntimeError("candidate kit library unavailable: " + str(KIT_IMPORT_ERROR))
    roots = [pathlib.Path(x) for x in os.environ.get("SZL_CORPUS_ROOTS", "").split(";") if x.strip()]
    home = pathlib.Path.home()
    roots += [p for p in home.iterdir() if p.is_dir() and p.name.lower().startswith("szl")] if home.is_dir() else []
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", "site-packages", "out", "outputs", ".cache",
            "appdata", "runs", "repos", "ledger", "dist", "build", ".tox", ".mypy_cache"}
    found, examined = [], 0
    for root in roots:
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in skip and not d.startswith(".")]
            if len(pathlib.Path(dirpath).parts) - base_depth > 5:
                dirnames[:] = []
                continue
            for fn in filenames:
                if not fn.lower().endswith(".jsonl"):
                    continue
                examined += 1
                if examined > 5000:
                    break
                p = pathlib.Path(dirpath) / fn
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size < 200 or size > 200 * 2**20:
                    continue
                sample, okc, rows, splits = [], 0, 0, Counter()
                try:
                    with open(p, "r", encoding="utf-8-sig") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            rows += 1
                            if len(sample) < 40:
                                sample.append(line)
                except OSError:
                    continue
                for line in sample:
                    try:
                        rec = kit.normalize_record(json.loads(line))
                        okc += 1
                        splits[rec["split"] or "untagged"] += 1
                    except (ValueError, TypeError):
                        pass
                if not sample or okc / len(sample) < 0.9:
                    continue
                found.append({"path": str(p), "dir": str(p.parent), "file": fn, "rows": rows, "bytes": size,
                              "sha256": kit.sha256_file(p), "mtime": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                              "sample_ok": f"{okc}/{len(sample)}", "split_tags": dict(splits),
                              "tokens": sorted(ntoks(str(p.relative_to(root)) if p.is_relative_to(root) else str(p)) - GENERIC)})
    wj(RUN / "local_corpora.json", found)
    log(f"corpora: {len(found)} candidate JSONL files under {[str(r) for r in roots]}")
    return found


def bind_corpus(name, corpora):
    """Pick the best local corpus group for a model by token overlap; group split files living in one directory."""
    mt = ntoks(name) - GENERIC
    excl = ("quarantine", "backup", "pre-fix", "prefix", "sample", "fixture", "example", "probe", "smoke")
    groups = {}
    for c in corpora:
        if any(x in c["file"].lower() for x in excl):
            continue
        score = len(mt & set(c["tokens"]))
        if score == 0:
            continue
        groups.setdefault(c["dir"], []).append((score, c))
    if not groups:
        return None
    def rank(c):
        fl = c["file"].lower()
        ver = tuple(int(x) for x in re.findall(r"\d+", (re.search(r"v(\d+(?:[._]\d+)*)", fl) or [None, ""])[1]))
        return (ver, int(any(x in fl for x in ("dedup", "clean", "final", "gated"))), c["rows"], c["mtime"])
    best_dir, best = max(groups.items(), key=lambda kv: (max(s for s, _ in kv[1]), max(rank(c) for _, c in kv[1])))
    files, alternatives = {}, []
    for score, c in sorted(best, key=lambda sc: rank(sc[1]), reverse=True):
        fl = c["file"].lower()
        split = ("train" if fl.startswith("train") else "dev" if re.match(r"(dev|eval|val)", fl) else
                 "test" if re.match(r"(test|held)", fl) else "adversarial" if re.match(r"(adversarial|red)", fl) else "all")
        if split in files:
            alternatives.append(c["path"])
        else:
            files[split] = {"path": c["path"], "sha256": c["sha256"], "rows": c["rows"], "split_tags": c["split_tags"]}
    if "all" in files and len(files) > 1:
        alternatives += [files.pop("all")["path"]]
    other_dirs = [d for d in groups if d != best_dir]
    return {"files": files, "alternatives": alternatives + other_dirs[:10], "dir": best_dir,
            "score": max(s for s, _ in best), "rows": sum(f["rows"] for f in files.values())}


# ------------------------------------------------------------------ candidates (the missing training code)
def hf_pin(repo):
    r = http(f"{HF}/api/models/{repo}")
    if not ok(r):
        return None, None
    j = r.json()
    card = j.get("cardData") if isinstance(j.get("cardData"), dict) else {}
    return j.get("sha"), card.get("license")


def pick_base(m):
    for b in m.get("adapter_base") or []:
        if "/" in b:
            return b, "ADAPTER_CONFIG"
    for b in m.get("base_model") or []:
        if "/" in b and not b.lower().startswith(HF_ORG.lower() + "/"):
            return b, "DECLARED_ON_CARD"
    n = m["model"].lower()
    small = any(x in n for x in ("nano", "mini", "0.5b", "0.6b", "0.8b", "tiny"))
    return ("Qwen/Qwen2.5-0.5B-Instruct" if small else "Qwen/Qwen2.5-1.5B-Instruct"), "DEFAULT_UNVERIFIED_CONFIRM_BEFORE_TRAINING"


def gen_candidate(m, corpora, est):
    name = m["model"].split("/")[-1]
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    cid = f"{slug}-candidate-{DATE}"
    folder = RUN / "training" / cid
    folder.mkdir(parents=True, exist_ok=True)
    base, sel = pick_base(m)
    base_sha, base_lic = hf_pin(base)
    bound = bind_corpus(name, corpora)
    rows = bound["rows"] if bound else 0
    full_steps = max(20, math.ceil(rows * 3 / 4)) if rows else 135
    td = {"origin": "LOCAL_OWNER_CORPUS" if bound else "UNBOUND", "binding_status": "BOUND_LOCAL" if bound else "UNBOUND",
          "binding_mode": "HEURISTIC_TOKEN_MATCH_LATEST_MODIFIED" if bound else None, "confirmed_by_owner": False,
          "rights": "OPERATOR_DECLARED" if bound else "UNDECLARED",
          "files": bound["files"] if bound else {}, "candidates_observed": (bound or {}).get("alternatives", []),
          "rights_boundary": "Only project-authored or rights-documented rows are admitted. Dev/test rows, Brain rows, OSINT rows, "
                             "third-party private data and other-model outputs are excluded from gradients.",
          "note": "Paths are the owner's local files; the digest binds the exact bytes. Nothing is committed to the repository."}
    cand = {"schema": "szl.frontier-model-candidate/v2", "candidate_id": cid, "state": "SOURCE_READY_NOT_TRAINED",
            "generated_by": f"szl-payload/{VERSION}", "generated_utc": now(), "target_repo_id": f"{HF_ORG}/{slug}-candidate-{DATE}",
            "predecessor": {"repo_id": m["model"], "revision": m.get("sha"), "observed_status": m["status"],
                            "role": "FROZEN_COMPARATOR_NOT_WEIGHT_INITIALIZATION"},
            "actual_training_base": {"repo_id": base, "revision": base_sha, "license": base_lic, "selection": sel, "load_in_4bit": True},
            "training_data": td,
            "training_recipe": {"smoke_optimizer_steps": 1, "full_optimizer_steps": full_steps, "per_device_batch_size": 1,
                                "gradient_accumulation_steps": 4, "max_length": 2048, "learning_rate": 0.0001, "warmup_steps": 10,
                                "optimizer": "adamw_8bit", "weight_decay": 0.01, "lr_scheduler": "constant_with_warmup", "seed": 11,
                                "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "response_only_loss": True,
                                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                "minimum_free_gpu_gib": 4.0, "export_levels": ["Q4_K_M", "Q5_K_M", "Q8_0"]},
            "runtime_lock": {},
            "evaluation_protocol": {"do_sample": False, "structured_max_new_tokens": 512, "refusal_max_new_tokens": 128,
                                    "required_strict_case_improvement_over_predecessor": 1,
                                    "claim_scope": "Project-authored, committed, preregistered suite evaluated by the author; "
                                                   "not a blind benchmark or independent certification."},
            "promotion_requirements": {"twelve_gates_pass": True, "owner_signed_training_envelope": True,
                                       "owner_signed_evaluation_envelope": True, "immutable_hub_byte_readback": True,
                                       "independent_inference_check": True, "unsigned_reports_are_receipt_eligible": False},
            "gates_file": "gates.json", "publication_eligible": False, "promotion": "NOT_PROMOTABLE",
            "source_repositories": m.get("sources", []), "family": m.get("family")}
    gates_text = json.dumps(GATES_DEFAULT, indent=2, sort_keys=True) + "\n"
    cand["gates_sha256"] = sha(gates_text)
    schema_file = next(iter(m.get("schema_files") or []), None)
    if schema_file:
        text = gt(f"{HF}/{m['model']}/raw/main/{schema_file}")
        if text:
            (folder / "output.schema.json").write_text(text, encoding="utf-8")
            cand["output_schema_file"] = "output.schema.json"
    (folder / "gates.json").write_text(gates_text, encoding="utf-8")
    wj(folder / "candidate.json", cand)
    repl = {"__VERSION__": VERSION, "__CANDIDATE_ID__": cid, "__DATE__": DATE, "__PREDECESSOR__": m["model"],
            "__OBSERVED_STATUS__": m["status"], "__TARGET__": cand["target_repo_id"]}
    for src in sorted(KIT_DIR.iterdir()):
        if src.is_file() and src.suffix in (".py", ".md"):
            text = src.read_text(encoding="utf-8")
            for k, v in repl.items():
                text = text.replace(k, v)
            (folder / src.name).write_text(text, encoding="utf-8")
    return {"candidate": cid, "model": m["model"], "status": m["status"], "base": base, "base_revision": base_sha or "UNPINNED",
            "base_selection": sel, "binding": td["binding_status"], "rows": rows, "corpus_dir": (bound or {}).get("dir"),
            "folder": str(folder), "repo_path": f"frontier/{cid}", "full_steps": full_steps}


def open_training_pr(cands, est):
    if not cands:
        return "NO_CANDIDATES"
    repo = next((r for r in est["github"] if r["name"].lower() == TRAIN_REPO.lower()), None)
    if not repo:
        return f"TRAINING_REPO_{TRAIN_REPO}_NOT_FOUND"
    full = f"{GH_ORG}/{repo['name']}"
    fp = sha("candidates|" + "|".join(sorted(c["candidate"] for c in cands)))
    if ledger_has(fp):
        return "PR_ALREADY_RECORDED_FOR_THIS_CANDIDATE_SET"
    code, o, _ = gh(["pr", "list", "--repo", full, "--state", "open", "--search", "frontier: add in:title training candidates in:title",
                     "--json", "number,title,url"])
    if code == 0:
        prior = [x for x in json.loads(o or "[]") if "training candidates" in str(x.get("title", ""))]
        if prior:
            return "PR_ALREADY_OPEN:" + str(prior[0].get("url"))
    if not OPEN_PR:
        return "PR_DRY_RUN"
    dest = repo_checkout(full)
    branch = f"frontier/candidates-{dt.datetime.now():%Y%m%d-%H%M%S}"
    base = repo["default_branch"] or "main"
    for args in (["fetch", "origin", base], ["checkout", "-B", branch, f"origin/{base}"]):
        code, out = git(dest, args)
        if code != 0:
            return "GIT_FAILED:" + out[-200:]
    added = []
    for c in cands:
        target = dest / c["repo_path"]
        if target.exists():
            c["pr_note"] = "EXISTS_IN_REPO_SKIPPED"
            continue
        shutil.copytree(c["folder"], target, ignore=shutil.ignore_patterns("out", "__pycache__", "*.pyc"))
        added.append(c["repo_path"])
    if not added:
        return "NOTHING_TO_ADD"
    index = ["# Frontier candidates generated " + DATE, "", "Generated by szl-payload " + VERSION + " from live estate discovery. "
             "Every folder is SOURCE_READY_NOT_TRAINED and fails closed until its curriculum is bound and leakage-gated. "
             "Presence here is not production admission.", "",
             mdt(["Candidate", "Predecessor", "Observed", "Base", "Binding", "Rows"],
                 [[c["candidate"], c["model"], c["status"], c["base"], c["binding"], c["rows"]] for c in cands if c["repo_path"] in added]), ""]
    (dest / "frontier" / f"CANDIDATES_{DATE}.md").write_text("\n".join(index), encoding="utf-8")
    git(dest, ["add", "frontier"])
    code, out = git(dest, ["commit", "-m", f"frontier: add {len(added)} fail-closed training candidates (SOURCE_READY_NOT_TRAINED)"], identity=True)
    if code != 0:
        return "COMMIT_FAILED:" + out[-200:]
    code, out = git(dest, ["push", "-u", "origin", branch])
    if code != 0:
        return "PUSH_FAILED:" + out[-300:]
    body = (RUN / "training" / "PR_BODY.md")
    body.write_text("\n".join(index + ["", "Each folder: candidate.json (exact base pin, frozen gates hash), qualify_runtime.py, curriculum.py "
                    "(digest + leakage gate), train_candidate.py (QLoRA, response-only loss, text-only tokenizer path, unsigned report), "
                    "evaluate_candidate.py (12-gate derivation, never writes PROMOTABLE), export_gguf.py (QUANT_MANIFEST + parity), "
                    "render_card.py (card from reports only), test_candidate_contract.py.", "",
                    "Review notes: base selections marked DEFAULT_UNVERIFIED need confirmation; local corpus bindings are "
                    "heuristic and need `confirmed_by_owner: true` before --full; nothing here publishes or promotes."]), encoding="utf-8")
    code, o, e = gh(["pr", "create", "--repo", full, "--base", base, "--head", branch,
                     "--title", f"frontier: add {len(added)} fail-closed training candidates ({DATE})", "--body-file", str(body)])
    if code != 0:
        return "PR_CREATE_FAILED:" + (o + e)[-300:]
    url = o.strip().splitlines()[-1] if o.strip() else "created"
    ledger_append({"action": "TRAINING_PR_OPENED", "repo": full, "branch": branch, "candidates": added, "ref": url, "fingerprint": fp})
    log(f"training PR opened: {url}")
    return url


def candidates(A, corpora, est):
    if kit is None:
        raise RuntimeError("candidate kit library unavailable: " + str(KIT_IMPORT_ERROR))
    if not KIT_DIR.is_dir() or not (KIT_DIR / "candidate_lib.py").is_file():
        raise RuntimeError(f"kit directory missing: {KIT_DIR}")
    todo = [m for m in A["models"] if m["sft_plan"] == "GENERATE_CANDIDATE"]
    out = []
    for m in todo:
        log(f"candidate kit for {m['model']} ({m['status']})")
        out.append(gen_candidate(m, corpora or [], est))
    pr = open_training_pr(out, est)
    for c in out:
        c["pr"] = pr
    (RUN / "training").mkdir(parents=True, exist_ok=True)
    wj(RUN / "training" / "candidates.json", out)
    log(f"candidates: {len(out)} generated; PR: {pr}")
    return out


# ------------------------------------------------------------------ train (owner metal)
def find_train_python():
    cands = [os.environ.get("SZL_TRAIN_PY")]
    home = pathlib.Path.home()
    if home.is_dir():
        for d in sorted(home.iterdir()):
            if d.is_dir() and d.name.lower().startswith("szl"):
                cands += [str(d / ".venv" / "Scripts" / "python.exe"), str(d / ".venv" / "bin" / "python")]
    probe = "import json,torch,transformers,peft;print(json.dumps({'cuda':torch.cuda.is_available(),'torch':torch.__version__}))"
    for c in cands:
        if not c or not pathlib.Path(c).is_file():
            continue
        p = subprocess.run([c, "-c", probe], capture_output=True, text=True, timeout=240)
        if p.returncode == 0 and p.stdout.strip().startswith("{"):
            info = json.loads(p.stdout.strip().splitlines()[-1])
            if info.get("cuda"):
                return c, info
    return None, None


def run_step(py, folder, script, args, timeout):
    env = dict(os.environ, SZL_CANDIDATE_OUT=str(folder / "out"), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    (folder / "out" / "logs").mkdir(parents=True, exist_ok=True)
    t = time.time()
    try:
        p = subprocess.run([py, "-u", script] + args, cwd=str(folder), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env, timeout=timeout)
        code, text = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        code, text = 124, f"TIMEOUT after {timeout}s\n" + str(e.stdout or "")[-2000:]
    (folder / "out" / "logs" / f"{script}.log").write_text(text, encoding="utf-8")
    return code, text[-600:], round(time.time() - t, 1)


def train_lane(cands):
    if TRAIN_MODE == "off":
        return [{"candidate": c["candidate"], "mode": "off", "outcome": "TRAIN_MODE_OFF"} for c in cands]
    py, info = find_train_python()
    out = []
    if not py:
        for c in cands:
            out.append({"candidate": c["candidate"], "mode": TRAIN_MODE, "outcome": "BLOCKED_NO_QUALIFIED_CUDA_ENV",
                        "detail": "no venv with torch+CUDA, transformers and peft found under ~/szl-*/.venv; set SZL_TRAIN_PY"})
        return out
    log(f"training python: {py} {info}")
    llama = os.environ.get("SZL_LLAMA_CPP")
    for c in cands:
        folder = pathlib.Path(c["folder"])
        row = {"candidate": c["candidate"], "mode": TRAIN_MODE, "python": py, "steps": []}
        if c["binding"] != "BOUND_LOCAL":
            row["outcome"] = "SKIPPED_UNBOUND_CURRICULUM"
            out.append(row)
            continue
        plan = [("qualify_runtime.py", [], 900), ("curriculum.py", [], 900),
                ("train_candidate.py", ["--smoke"] if TRAIN_MODE == "smoke" else ["--full"] + (["--confirm-binding"] if CONFIRM else []),
                 2400 if TRAIN_MODE == "smoke" else 6 * 3600),
                ("evaluate_candidate.py", ["--split", "dev"], 3 * 3600)]
        if llama:
            plan.append(("export_gguf.py", ["--llama-cpp", llama], 3 * 3600))
        plan.append(("render_card.py", [], 300))
        row["outcome"] = "COMPLETED"
        for script, args, timeout in plan:
            log(f"[{c['candidate']}] {script} {' '.join(args)}")
            code, tail, secs = run_step(py, folder, script, args, timeout)
            row["steps"].append({"step": script, "exit": code, "seconds": secs})
            if code != 0:
                row["outcome"] = f"STOPPED_AT_{script}_EXIT_{code}"
                row["detail"] = tail
                break
        for name in ("training_report.json", "evaluation_report.json", "gate_results.json"):
            p = folder / "out" / name
            if p.is_file():
                j = json.loads(p.read_text(encoding="utf-8"))
                row[name.split(".")[0]] = {k: j.get(k) for k in ("state", "finalTrainLoss", "adapterSha256", "release_gate", "gate_verdict", "promotion") if k in j}
        out.append(row)
    return out


# ------------------------------------------------------------------ reports
def reports(est, A, Q, P, R, C, T):
    M = A.get("models", [])
    for name, rows in (("models", M), ("datasets", A.get("datasets", [])), ("spaces", A.get("spaces", [])),
                       ("github", A.get("repos", [])), ("drift", A.get("drift", [])), ("sync", A.get("sync", [])),
                       ("quarantine", Q), ("pull_requests", P), ("pr_repairs", R), ("candidates", C), ("training_runs", T)):
        wcsv(RUN / f"{name}.csv", rows)
    counts = {"GitHub repositories": len(est["github"]), "GitHub archived": sum(r["archived"] for r in est["github"]),
              "HF models": len(est["models"]), "HF datasets": len(est["datasets"]), "HF spaces": len(est["spaces"]),
              f"{TRAIN_REPO} source bindings": len((est.get("forge") or {}).get("bindings", {}).get("artifacts", []))}
    scored = sorted([m for m in M if m["score"] is not None], key=lambda m: m["score"])
    drift = [d for d in A.get("drift", []) if d["flags"]]
    gov = [r for r in A.get("repos", []) if r["governance_gaps"]]
    broken = [s for s in A.get("spaces", []) if s["broken"]]
    prc = Counter(str(p.get("result")) for p in P)
    mode = (f"merge={'ON' if MERGE else 'DRY-RUN'} | repair={'ON' if REPAIR else 'DRY-RUN'} | HF quarantine PRs={'ON' if EXECUTE else 'DRY-RUN'} | "
            f"training PR={'ON' if OPEN_PR else 'DRY-RUN'} | train={TRAIN_MODE}")
    md = ["# SZL Estate Scorecard", "", f"Generated {now()} | payload {VERSION} | {mode}", "",
          "## Live estate", "", mdt(["Asset", "Count"], list(counts.items())), "",
          "## Artifact status", "", mdt(["Status", "Count"], sorted(Counter(m["status"] for m in M).items())), "",
          "## Source plans (what trains each Hub model)", "", mdt(["Plan", "Count"], sorted(Counter(m["sft_plan"] for m in M).items())), "",
          mdt(["Model", "Status", "Plan", f"{TRAIN_REPO} folders", "Trainer", "Evaluator", "Receipts", "Gaps", "Next action"],
              [[m["model"], m["status"], m["sft_plan"], ", ".join(m["forge_folders"][:3]), ", ".join(m["forge_trainer"][:2]),
                ", ".join(m["forge_evaluator"][:2]), m["forge_receipts"], ", ".join(m["forge_gaps"]), m["next_action"]]
               for m in sorted(M, key=lambda x: (x["sft_plan"], x["model"]))]), "",
          "## Tiers", "", mdt(["Tier", "Count"], sorted(Counter(m["tier"] for m in M).items())), "",
          "Score = weights 30 + card 15 + license 5 + lineage 10 + source trainer 20 + receipts 10 + quant integrity 10. A>=80, B>=60, C>=40, D<40.", "",
          "## Model scorecard (weakest first)", "",
          mdt(["Model", "Score", "Tier", "Status", "Family", "Flags", "Next action"],
              [[m["model"], m["score"], m["tier"], m["status"], m["family"], ", ".join(m["flags"]), m["next_action"]] for m in scored]), "",
          f"## Training candidates generated ({len(C)})", "",
          mdt(["Candidate", "Predecessor", "Observed", "Base", "Selection", "Binding", "Rows", "PR"],
              [[c["candidate"], c["model"], c["status"], c["base"], c["base_selection"], c["binding"], c["rows"], c.get("pr")] for c in C]), "",
          f"## Training runs on this machine ({len(T)})", "",
          mdt(["Candidate", "Mode", "Outcome", "Steps", "Training", "Evaluation"],
              [[t["candidate"], t["mode"], t.get("outcome"), " > ".join(f"{s['step']}:{s['exit']}" for s in t.get("steps", [])),
                json.dumps(t.get("training_report", {})), json.dumps(t.get("evaluation_report", {}))] for t in T]), "",
          f"## Quantization drift ({len(drift)})", "",
          mdt(["Repo", "Levels", "mmproj", "Flags"], [[d["model"], d["levels"], d["mmproj"], ", ".join(d["flags"])] for d in drift]), "",
          f"## Quarantine ({len(Q)})", "", mdt(["Type", "Repo", "Reason", "Action"], [[q["repo_type"], q["repo_id"], q["reason"], q["action"]] for q in Q]), "",
          f"## Source/artifact sync ({len(A.get('sync', []))})", "", mdt(["Asset", "Issue", "Detail"], [[s["asset"], s["issue"], s["detail"]] for s in A.get("sync", [])]), "",
          f"## Broken Spaces ({len(broken)})", "", mdt(["Space", "Stage", "SDK"], [[s["space"], s["stage"], s["sdk"]] for s in broken]), "",
          f"## Governance gaps ({len(gov)} active repos)", "", mdt(["Repo", "Missing"], [[r["repo"], ", ".join(r["governance_gaps"])] for r in gov]), "",
          "## Pull requests", "", mdt(["Result", "Count"], sorted(prc.items())), "",
          f"## PR repairs ({len(R)})", "",
          mdt(["PR", "Was", "Classes", "Action", "Detail"], [[f"{r['repo']}#{r['number']}", r.get("pr_result"), ", ".join(r.get("classes") or []),
                                                             r.get("action"), (r.get("detail") or r.get("brief") or r.get("comment") or "")[:120]] for r in R]), "",
          "## Lanes", "", mdt(["Lane", "Status", "Seconds", "Error"], [[ln["lane"], ln["status"], ln["seconds"], ln.get("error", "")] for ln in LANES])]
    (RUN / "SCORECARD.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    briefs = [r for r in R if r.get("brief")]
    repairs = ["# Codex brief: pull requests that need code judgment", "", f"Generated {now()} by payload {VERSION}. Each item links a local brief with "
               "redacted failing-log tails. Fix root causes; never weaken, skip or delete a check; proof and doctrine gates that are red by design stay red.", ""]
    for r in briefs:
        repairs += [f"## {r['repo']}#{r['number']} - {r.get('title')}", "", f"- URL: {r.get('url')}", f"- Classification: {', '.join(r.get('classes') or [])}",
                    f"- Payload action: {r.get('action')}", f"- Brief: `{r['brief']}`", ""]
    (RUN / "CODEX-PR-REPAIRS.md").write_text("\n".join(repairs) + "\n", encoding="utf-8")

    runbook = ["# Owner GPU runbook", "", f"Generated {now()}. Exact commands, in order, per generated candidate. Run in a qualified venv "
               "(torch+CUDA, transformers, peft; unsloth optional). Smoke proves the path; full needs a confirmed binding.", ""]
    for c in C:
        runbook += [f"## {c['candidate']}", "", f"Folder: `{c['folder']}` | predecessor `{c['model']}` | base `{c['base']}` ({c['base_selection']}) | "
                    f"binding {c['binding']} ({c['rows']} rows)", "", "```powershell", f"Set-Location -LiteralPath '{c['folder']}'",
                    "python qualify_runtime.py", "python curriculum.py", "python train_candidate.py --smoke",
                    "python train_candidate.py --full --confirm-binding", "python evaluate_candidate.py --split dev",
                    "python evaluate_candidate.py --split test", "python export_gguf.py --llama-cpp <path-to-llama.cpp>", "python render_card.py",
                    "python -m pytest -q test_candidate_contract.py", "```", ""]
    (RUN / "OWNER-GPU-RUNBOOK.md").write_text("\n".join(runbook) + "\n", encoding="utf-8")

    has_trainer = [m for m in M if m["sft_plan"] == "HAS_TRAINER"]
    codex = f"""# SZL Holdings Codex v4 - Estate completion, PR repair, and training program

Generated {now()} by payload {VERSION}. All numbers come from this run's live discovery. Read `SCORECARD.md`,
`CODEX-PR-REPAIRS.md`, `OWNER-GPU-RUNBOOK.md` and the CSVs first. Do not trust historical counts.

## Live estate

{mdt(["Asset", "Count"], list(counts.items()))}

## Doctrine (non-negotiable)

1. No fabricated metrics, evidence, provenance, runtime state, or novelty claims.
2. No training on synthetic-fixture, unlicensed, or untraceable data; local owner corpora carry rights=OPERATOR_DECLARED until documented.
3. No promotion without a fresh passing twelve-gate receipt plus owner-signed envelopes and independent inference check.
4. No red, pending, draft, or HOLD merges. Never bypass branch protection, signatures, or required checks.
5. Never overwrite a public release; every candidate targets a new repository id with the predecessor as frozen comparator.
6. Label every statement FACT, INFERENCE, PROPOSAL, or BLOCKED. A blocked lane is a remediation item, not a stop.

## Source map: where each Hub model is trained

{mdt(["Plan", "Count"], sorted(Counter(m["sft_plan"] for m in M).items()))}

Models with an existing trainer (do not add a second one; close their gaps in place):

{mdt(["Model", "Folders", "Gaps"], [[m["model"], ", ".join(m["forge_folders"][:3]), ", ".join(m["forge_gaps"]) or "none observed"] for m in has_trainer])}

## Generated candidates ({len(C)})

Each candidate folder is a complete fail-closed kit (qualify -> curriculum/leakage -> train -> evaluate -> export -> card -> tests)
in the frontier-candidate idiom. It trains an isolated challenger into a NEW target id; the predecessor is a frozen comparator.
UNBOUND candidates need a project-authored or rights-documented curriculum before any GPU time.

{mdt(["Candidate", "Predecessor", "Base @ revision", "Binding", "Rows"], [[c["candidate"], c["model"], f"{c['base']} @ {c['base_revision']}", c["binding"], c["rows"]] for c in C])}

## Training runs observed on this machine ({len(T)})

A smoke run proves the path on owner metal; it is not a challenger. A full run yields TRAINED_CHALLENGER_UNSIGNED;
sign with the owner signer, then evaluate on the preregistered test split with --comparator before any publication decision.

{mdt(["Candidate", "Mode", "Outcome", "Release gate"], [[t["candidate"], t["mode"], t.get("outcome"), (t.get("evaluation_report") or {}).get("release_gate")] for t in T])}

## PR repairs ({len(R)})

Deterministic repairs ran where safe (flaky reruns, Dependabot rebase/recreate, formatter pushes, base-merge pushes).
Everything else has a brief in `CODEX-PR-REPAIRS.md`; intentional red proof gates are preserved.

{mdt(["PR", "Classes", "Action"], [[f"{r['repo']}#{r['number']}", ", ".join(r.get("classes") or []), r.get("action")] for r in R])}

## Twelve release gates

Rights/provenance; leakage isolation; schema validity; evidence-handle validity; calibration (ECE); abstention and false-abstention;
policy compliance; red-team/prompt-injection; regression vs parent (strict beat, ties fail); full-precision vs quant parity;
reproducibility; model-card claims equal receipts. Thresholds are frozen by hash in each candidate before results exist.

## Exit condition

Done when every Hub model is weighted and gate-verified, explicitly experimental with listed failing gates, classified as software,
quarantined, skipped by receipt, or honestly retired, and a fresh discovery run reproduces that state with zero new gaps.
"""
    (RUN / "CODEX-v4.md").write_text(codex, encoding="utf-8")
    wj(RUN / "summary.json", {"generated_utc": now(), "version": VERSION, "counts": counts, "plans": dict(Counter(m["sft_plan"] for m in M)),
                              "candidates": len(C), "training_runs": [{"candidate": t["candidate"], "outcome": t.get("outcome")} for t in T],
                              "pr_results": dict(prc), "pr_repairs": dict(Counter(str(r.get("action")) for r in R)),
                              "quarantine": len(Q), "drift": len(drift), "ledger": ledger_verify()})


def main():
    log(f"SZL payload {VERSION} | org={GH_ORG} hf={HF_ORG} train_repo={TRAIN_REPO} | merge={MERGE} repair={REPAIR} execute_hf={EXECUTE} "
        f"clone={CLONE} open_training_pr={OPEN_PR} train={TRAIN_MODE} kit={KIT_DIR}")
    try:
        est = lane("discover", discover)
        if est is None:
            return 2
        A = lane("analyze", analyze, est) or {}
        Q = (lane("quarantine", quarantine, A) if A else None) or []
        P = lane("pull_requests", pr_lane, est) or []
        R = lane("pr_repair", pr_repair, P) or []
        corpora = lane("corpora", scan_corpora) or []
        C = (lane("candidates", candidates, A, corpora, est) if A else None) or []
        T = lane("train", train_lane, C) or []
        lane("reports", reports, est, A, Q, P, R, C, T)
    finally:
        wj(RUN / "lanes.json", LANES)
        (ROOT / "LATEST.txt").write_text(str(RUN), encoding="utf-8")
        log(f"Run folder: {RUN}")
    return 1 if any(ln["status"] == "BLOCKED" for ln in LANES) else 0


if __name__ == "__main__":
    sys.exit(main())
