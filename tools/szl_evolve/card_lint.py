#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model/dataset card linter (LOCAL PROPOSAL). Read-only. Rule ids are stable (CL###).
Checks what Hub docs recommend (license, base_model, datasets, sections), what Croissant RAI asks
(collection, bias, limitations, personal/sensitive info) and the estate's own honesty failures
(unsupported 'independently evaluated'/'immutable pins' claims, license vs base-license mismatch).
"""
from __future__ import annotations
import argparse, json, os, re, sys
from typing import Dict, List, Tuple
import yaml

RESTRICTED_BASES = {  # base id (lowercase) -> (license class, note). Source: pukyu REPORT.md A20/A26 + NVIDIA OML notice finding F4.
    "qwen/qwen2.5-3b": ("research_only", "Qwen Research License (non-commercial)"),
    "qwen/qwen2.5-3b-instruct": ("research_only", "Qwen Research License (non-commercial)"),
    "qwen/qwen2.5-coder-3b": ("research_only", "Qwen Research License (non-commercial)"),
    "qwen/qwen2.5-coder-3b-instruct": ("research_only", "Qwen Research License (non-commercial)"),
}
PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "cc-by-4.0", "cc0-1.0"}
CLAIMS = re.compile(r"(?i)\b(independently evaluated|independently verified|every pin (?:is )?immutable|state[- ]of[- ]the[- ]art|production[- ]ready|zero contamination|no contamination|fully reproducible|guaranteed)\b")
PLACEHOLDER = re.compile(r"(?i)\[More Information Needed\]|\bTODO\b|\bTBD\b|lorem ipsum")
SECTION_RX = {"limitations": re.compile(r"(?im)^#+\s*.*(limitation|out[- ]of[- ]scope|known issues|caveat)"),
              "intended_use": re.compile(r"(?im)^#+\s*.*(intended use|uses?\b|usage)"),
              "bias_risk": re.compile(r"(?im)^#+\s*.*(bias|risk|safety|ethic)"),
              "provenance": re.compile(r"(?im)^#+\s*.*(provenance|source|lineage|how (?:it|this) was (?:built|generated|created)|data collection|training data)")}
SECRET = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bAKIA[0-9A-Z]{16}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----")
HEX_PIN = re.compile(r"\b[0-9a-f]{7,40}\b")
FM = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)


def parse(text: str) -> Tuple[Dict, str, str]:
    m = FM.match(text)
    if not m:
        return {}, text, "missing"
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        return (meta if isinstance(meta, dict) else {}), text[m.end():], ("ok" if isinstance(meta, dict) else "not_a_mapping")
    except yaml.YAMLError:
        return {}, text[m.end():], "yaml_error"


def lint(text: str, kind: str = "model", name: str = "") -> Dict:
    meta, body, fm = parse(text)
    f: List[Dict] = []
    add = lambda code, sev, msg: f.append({"code": code, "severity": sev, "message": msg})
    if fm != "ok":
        add("CL001", "ERROR", f"front matter {fm}")
    lic = meta.get("license")
    if not lic:
        add("CL002", "ERROR", "no license in metadata")
    elif str(lic).lower() == "other" and not (meta.get("license_name") or meta.get("license_link")):
        add("CL003", "ERROR", "license: other without license_name/license_link")
    base = meta.get("base_model")
    bases = [base] if isinstance(base, str) else (base or [])
    if kind == "model":
        tags = [str(t).lower() for t in (meta.get("tags") or [])]
        ft_hint = any(t in tags for t in ("lora", "peft", "sft", "dpo", "adapter", "fine-tuned", "trl")) or str(meta.get("library_name", "")).lower() == "peft" \
            or re.search(r"(?i)sft|dpo|lora|adapter|-r\d\b", name or "")
        if ft_hint and not bases:
            add("CL004", "ERROR", "fine-tune/adapter without base_model")
        for b in bases:
            hit = RESTRICTED_BASES.get(str(b).lower())
            if hit and lic and str(lic).lower() in PERMISSIVE:
                add("CL005", "ERROR", f"license {lic} but base {b} is under {hit[1]}")
            if str(b).lower().startswith("nvidia/") and not re.search(r"(?i)nvidia open model license|NOTICE", text):
                add("CL006", "WARN", f"base {b} (NVIDIA) but card has no NVIDIA Open Model License notice")
        if not meta.get("datasets") and re.search(r"(?i)\btrained on\b|\bfine-?tuned on\b", body):
            add("CL007", "WARN", "says trained on data but metadata has no datasets list")
    if kind == "dataset" and not (meta.get("pretty_name") or re.search(r"(?m)^#\s+\S", body)):
        add("CL008", "WARN", "no pretty_name or title")
    for key, rx in SECTION_RX.items():
        if not rx.search(body):
            add("CL010", "WARN" if key != "limitations" else "ERROR", f"no section for {key}")
    if kind == "dataset" and not re.search(r"(?i)personal|pii|sensitive|privacy|anonymi[sz]", body):
        add("CL011", "ERROR", "dataset card never states personal/sensitive-information status (Croissant rai:personalSensitiveInformation)")
    if (meta.get("datasets") or kind == "dataset") and not HEX_PIN.search(body):
        add("CL012", "WARN", "no commit/revision pin (hex) anywhere in card body")
    for m in CLAIMS.finditer(body):
        add("CL020", "WARN", f"unsupported-claim phrase '{m.group(0)}' (needs evidence link or removal)")
    if re.search(r"(?i)\b(accuracy|pass@1|score|abstain\w*)\b[^.\n]{0,40}\d", body) and not re.search(r"(?i)\bn\s*=\s*\d+|\b\d+\s*(?:probes|samples|cases|examples|rows)\b", body):
        add("CL021", "WARN", "metric reported without sample size")
    if PLACEHOLDER.search(body):
        add("CL030", "WARN", "placeholder text left in card")
    if SECRET.search(text):
        add("CL040", "ERROR", "secret-looking value in card (not shown)")
    err = sum(1 for x in f if x["severity"] == "ERROR")
    return {"name": name, "kind": kind, "errors": err, "warnings": len(f) - err, "findings": f}


def lint_dir(card_dir: str) -> Dict:
    rows = []
    for fn in sorted(os.listdir(card_dir)):
        if not fn.endswith(".md"):
            continue
        kind = "dataset" if fn.startswith("datasets__") else ("space" if fn.startswith("spaces__") else "model")
        if kind == "space":
            continue
        nm = fn[:-3].split("__", 2)[-1]
        rows.append(lint(open(os.path.join(card_dir, fn), encoding="utf-8", errors="replace").read(), kind, nm))
    by: Dict[str, int] = {}
    for r in rows:
        for x in r["findings"]:
            by[x["code"]] = by.get(x["code"], 0) + 1
    return {"cards": len(rows), "cards_with_errors": sum(1 for r in rows if r["errors"]), "by_code": dict(sorted(by.items())), "results": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path"); ap.add_argument("--kind", default="model", choices=["model", "dataset"])
    a = ap.parse_args(argv)
    if os.path.isdir(a.path):
        rep = lint_dir(a.path); print(json.dumps({k: rep[k] for k in ("cards", "cards_with_errors", "by_code")}, indent=2))
        return 1 if rep["cards_with_errors"] else 0
    rep = lint(open(a.path, encoding="utf-8").read(), a.kind, os.path.basename(a.path)); print(json.dumps(rep, indent=2))
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
