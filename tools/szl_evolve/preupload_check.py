#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-upload privacy / pickle / secret check (LOCAL PROPOSAL). Read-only. Uploads nothing. Never loads a pickle.

Mirrors what the Hub does after the fact (pickle import listing, ClamAV on commit, TruffleHog on push) but
runs BEFORE bytes leave the box, and adds PII patterns and a private-by-default visibility rule.
Findings never contain the matched value, only file, line, kind.
"""
from __future__ import annotations
import argparse, io, json, os, pickletools, re, sys, zipfile
from typing import Dict, List, Optional, Tuple

PICKLE_EXT = {".pkl", ".pickle", ".pt", ".pth", ".bin", ".ckpt", ".joblib"}
TEXT_EXT = {".jsonl", ".json", ".md", ".txt", ".csv", ".tsv", ".yaml", ".yml", ".py", ".sh", ".cfg", ".toml", ".log"}
# modules that give code execution / I/O when unpickled
DANGEROUS_MODULES = {"os", "posix", "nt", "subprocess", "sys", "socket", "shutil", "importlib", "pty", "runpy",
                     "marshal", "ctypes", "builtins", "__builtin__", "webbrowser", "requests", "urllib", "urllib.request",
                     "http", "http.client", "ftplib", "smtplib", "pickle", "dill", "cloudpickle", "code", "codeop",
                     "commands", "timeit", "multiprocessing", "asyncio", "tempfile"}
DANGEROUS_BUILTINS = {"eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "input", "breakpoint"}
SAFE_GLOBALS = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
                ("torch", "FloatStorage"), ("torch", "HalfStorage"), ("torch", "BFloat16Storage"), ("torch", "LongStorage"),
                ("torch", "IntStorage"), ("torch", "BoolStorage"), ("torch", "Size"), ("torch", "float16"), ("torch", "float32"),
                ("torch", "bfloat16"), ("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}

SECRET_PATTERNS = {
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer_literal": re.compile(r"(?i)\bauthorization\s*[:=]\s*['\"]?bearer\s+[A-Za-z0-9._\-]{20,}"),
    "sk_style_key": re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
}
SSN = re.compile(r"(?<![\d-])(?!000|666|9\d\d)(\d{3})-(?!00)(\d{2})-(?!0000)(\d{4})(?![\d-])")
CARD = re.compile(r"(?<![\w.-])(?:\d[ -]?){12,18}\d(?![\w.-])")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "users.noreply.github.com"}


def card_brand_ok(d: str) -> bool:
    """Issuer-prefix + length check, so random long digit runs (ids, timestamps, hash digits) are not flagged."""
    n = len(d)
    if d[0] == "4":
        return n in (13, 16, 19)
    if re.match(r"5[1-5]|2(?:2[2-9][1-9]|2[3-9]\d|[3-6]\d\d|7[01]\d|720)", d):
        return n == 16
    if d[:2] in ("34", "37"):
        return n == 15
    if d[:4] == "6011" or d[:2] == "65" or d[:3] in tuple(str(x) for x in range(644, 650)):
        return n in (16, 19)
    return False


def luhn(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            d = d - 9 if d > 9 else d
        s += d
        alt = not alt
    return s % 10 == 0


def scan_pickle_bytes(data: bytes) -> Dict:
    """Statically list imports via pickletools.genops. Never executes anything."""
    imports: List[Tuple[str, str]] = []
    strs: List[str] = []
    reduce_ops = 0
    try:
        for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
            n = op.name
            if n in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8", "STRING", "BINSTRING", "SHORT_BINSTRING"):
                strs.append(arg if isinstance(arg, str) else str(arg))
            elif n == "GLOBAL":
                m, _, a = str(arg).partition(" ")
                imports.append((m, a))
            elif n == "STACK_GLOBAL" and len(strs) >= 2:
                imports.append((strs[-2], strs[-1]))
            elif n in ("REDUCE", "NEWOBJ", "NEWOBJ_EX", "BUILD", "INST", "OBJ"):
                reduce_ops += 1
    except Exception as e:  # truncated or not a pickle
        return {"parse_error": type(e).__name__, "imports": imports, "verdict": "UNPARSEABLE"}
    dangerous = [f"{m}.{a}" for m, a in imports
                 if m.split(".")[0] in {x.split(".")[0] for x in DANGEROUS_MODULES}
                 or (m in ("builtins", "__builtin__") and a in DANGEROUS_BUILTINS)]
    unreviewed = sorted({f"{m}.{a}" for m, a in imports if (m, a) not in SAFE_GLOBALS} - set(dangerous))
    verdict = "DANGEROUS" if dangerous else ("UNREVIEWED_IMPORTS" if unreviewed else "ALLOWLISTED_ONLY")
    return {"imports": sorted({f"{m}.{a}" for m, a in imports}), "dangerous": sorted(set(dangerous)),
            "unreviewed": unreviewed, "reduce_like_ops": reduce_ops, "verdict": verdict}


def scan_pickle_file(path: str) -> Dict:
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":  # torch zip: data.pkl inside
        results = []
        try:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.endswith(".pkl"):
                        results.append((name, scan_pickle_bytes(z.read(name))))
        except zipfile.BadZipFile:
            return {"verdict": "UNPARSEABLE", "parse_error": "BadZipFile"}
        if not results:
            return {"verdict": "ZIP_WITHOUT_PICKLE"}
        worst = sorted(results, key=lambda r: ["ALLOWLISTED_ONLY", "UNREVIEWED_IMPORTS", "UNPARSEABLE", "DANGEROUS"].index(r[1]["verdict"]))[-1][1]
        return {**worst, "members": [n for n, _ in results]}
    with open(path, "rb") as f:
        return scan_pickle_bytes(f.read())


def scan_text_file(path: str, max_bytes: int = 64 * 1024 * 1024) -> List[Dict]:
    out: List[Dict] = []
    size = os.path.getsize(path)
    if size > max_bytes:
        return [{"kind": "SCAN_SKIPPED_TOO_LARGE", "line": 0, "severity": "WARN"}]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            for kind, rx in SECRET_PATTERNS.items():
                if rx.search(line):
                    out.append({"kind": f"SECRET:{kind}", "line": ln, "severity": "ERROR"})
            if SSN.search(line):
                out.append({"kind": "PII:ssn_format", "line": ln, "severity": "ERROR"})
            for m in CARD.finditer(line):
                d = re.sub(r"\D", "", m.group(0))
                if card_brand_ok(d) and luhn(d) and len(set(d)) > 2:
                    out.append({"kind": "PII:card_luhn", "line": ln, "severity": "ERROR"})
            for m in EMAIL.finditer(line):
                if m.group(0).split("@")[1].lower() not in ALLOWED_EMAIL_DOMAINS:
                    out.append({"kind": "PII:email", "line": ln, "severity": "WARN"})
    return out


def check_dir(root: str, target_visibility: str = "private", public_approval_ref: Optional[str] = None,
              allow_pickle: bool = False) -> Dict:
    findings: List[Dict] = []
    if target_visibility != "private" and not public_approval_ref:
        findings.append({"file": "*", "kind": "PUBLIC_WITHOUT_APPROVAL", "severity": "ERROR", "line": 0})
    files = 0
    for dp, _dn, fns in os.walk(root):
        if ".git" in dp.split(os.sep):
            continue
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            files += 1
            ext = os.path.splitext(fn)[1].lower()
            if ext in PICKLE_EXT:
                r = scan_pickle_file(p)
                if r["verdict"] == "UNPARSEABLE" and ext in (".bin",):
                    r["verdict"] = "NOT_A_PICKLE_OR_OPAQUE"
                sev = "ERROR" if r["verdict"] in ("DANGEROUS", "UNPARSEABLE") else ("ERROR" if not allow_pickle and r["verdict"] != "NOT_A_PICKLE_OR_OPAQUE" else "WARN")
                findings.append({"file": rel, "kind": f"PICKLE:{r['verdict']}", "severity": sev, "line": 0,
                                 "detail": {k: r[k] for k in ("dangerous", "unreviewed") if r.get(k)}})
            if ext in (".safetensors", ".gguf"):
                continue
            if ext in TEXT_EXT or ext == "":
                for f in scan_text_file(p):
                    findings.append({"file": rel, **f})
    errors = sum(1 for f in findings if f["severity"] == "ERROR")
    return {"schema": "szl.preupload.check/v1", "root": root, "files_scanned": files, "target_visibility": target_visibility,
            "verdict": "BLOCK" if errors else "PASS_NOT_UPLOADED", "errors": errors,
            "warnings": len(findings) - errors, "findings": findings,
            "limits": "regex PII/secrets only; unreviewed pickle imports need human review; passing never authorizes an upload"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root"); ap.add_argument("--target-visibility", default="private", choices=["private", "public"])
    ap.add_argument("--public-approval-ref"); ap.add_argument("--allow-pickle", action="store_true")
    a = ap.parse_args(argv)
    rep = check_dir(a.root, a.target_visibility, a.public_approval_ref, a.allow_pickle)
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] == "PASS_NOT_UPLOADED" else 1


if __name__ == "__main__":
    sys.exit(main())
