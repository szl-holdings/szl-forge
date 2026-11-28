#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned training-run manifest + fail-closed validator (LOCAL PROPOSAL; nothing here launches, trains or uploads).

Why: the estate audit (REPORT.md section 3/3c) found no runner that records base-model
revision, data revision+sha, code commit and a cost cap; salad_worker.py silently
truncates to 2,000 rows and creates public Hub repos by default. Leaders pin inputs
(datatrove logs per-task completions/stats; OLMo publishes checkpoints + configs; SLSA
treats training as a build with data/models as dependencies).

A VALID manifest is necessary, never sufficient: it does not authorize a launch.
Spend/train/publish still need Stephen's direct approval, referenced by approval_ref.
"""
from __future__ import annotations
import argparse, json, re, sys
from typing import Any, Dict, List

SCHEMA = "szl.train.run-manifest/v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "cc0-1.0"}
RESTRICTED_BASE_CLASSES = {"research_only", "custom_noted"}  # Qwen Research License; NVIDIA Open Model License
SECRET_LIKE = re.compile(r"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def _f(code: str, severity: str, path: str, msg: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "path": path, "message": msg}


def _req(d: Any, key: str, path: str, out: List[Dict[str, str]], typ=None):
    if not isinstance(d, dict) or key not in d or d[key] in (None, ""):
        out.append(_f("MISSING", "ERROR", f"{path}.{key}", "required field missing or empty"))
        return None
    if typ and not isinstance(d[key], typ):
        out.append(_f("TYPE", "ERROR", f"{path}.{key}", f"expected {typ}"))
        return None
    return d[key]


def validate(m: Dict[str, Any]) -> Dict[str, Any]:
    out: List[Dict[str, str]] = []
    if m.get("schema") != SCHEMA:
        out.append(_f("SCHEMA", "ERROR", "schema", f"must equal {SCHEMA}"))
    # provenance honesty: reconstructed manifests are allowed but labelled
    recorded = m.get("pins_recorded_at_launch")
    if recorded is not True:
        out.append(_f("NOT_RECORDED_AT_LAUNCH", "WARN", "pins_recorded_at_launch",
                      "manifest was reconstructed post hoc (or flag missing); it cannot authorize a launch"))
    # base model
    bm = _req(m, "base_model", "", out, dict)
    if bm is not None:
        _req(bm, "id", "base_model", out)
        rev = _req(bm, "revision", "base_model", out)
        if rev and not HEX40.match(str(rev)):
            out.append(_f("UNPINNED_REVISION", "ERROR", "base_model.revision", "must be a full 40-hex commit sha (no branch, tag or short sha)"))
        _req(bm, "license_class", "base_model", out)
    # datasets
    ds = _req(m, "datasets", "", out, list)
    roles = set()
    for i, d in enumerate(ds or []):
        p = f"datasets[{i}]"
        _req(d, "id", p, out)
        r = _req(d, "revision", p, out)
        if r and not HEX40.match(str(r)):
            out.append(_f("UNPINNED_REVISION", "ERROR", f"{p}.revision", "must be a full 40-hex commit sha"))
        s = _req(d, "sha256", p, out)
        if s and not HEX64.match(str(s)):
            out.append(_f("BAD_SHA256", "ERROR", f"{p}.sha256", "must be 64 lowercase hex"))
        rows, used = d.get("rows"), d.get("rows_used")
        if not isinstance(rows, int) or not isinstance(used, int):
            out.append(_f("MISSING", "ERROR", f"{p}.rows/rows_used", "integer rows and rows_used required"))
        elif used != rows and not d.get("truncation_reason"):
            out.append(_f("SILENT_TRUNCATION", "ERROR", p, f"rows_used {used} != rows {rows} with no truncation_reason"))
        role = d.get("role")
        if role not in ("train", "validation", "holdout"):
            out.append(_f("BAD_ROLE", "ERROR", f"{p}.role", "role must be train|validation|holdout"))
        roles.add(role)
        if not d.get("license"):
            out.append(_f("MISSING", "ERROR", f"{p}.license", "dataset license required"))
    if ds is not None and "train" not in roles:
        out.append(_f("NO_TRAIN_DATA", "ERROR", "datasets", "no dataset with role=train"))
    if ds is not None and "holdout" not in roles:
        out.append(_f("NO_HOLDOUT", "ERROR", "datasets", "no sealed holdout declared (contamination gate cannot run)"))
    # code
    code = _req(m, "code", "", out, dict)
    if code is not None:
        _req(code, "repo", "code", out)
        c = _req(code, "commit", "code", out)
        if c and not HEX40.match(str(c)):
            out.append(_f("UNPINNED_REVISION", "ERROR", "code.commit", "must be a full 40-hex commit sha"))
        es = _req(code, "entrypoint_sha256", "code", out)
        if es and not HEX64.match(str(es)):
            out.append(_f("BAD_SHA256", "ERROR", "code.entrypoint_sha256", "must be 64 lowercase hex"))
    # budget
    b = _req(m, "budget", "", out, dict)
    if b is not None:
        for k in ("max_usd", "max_hours"):
            v = b.get(k)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
                out.append(_f("NO_COST_CAP", "ERROR", f"budget.{k}", "positive number required"))
        _req(b, "stop_rule", "budget", out)
        _req(b, "approval_ref", "budget", out)
    # output / privacy
    o = _req(m, "output", "", out, dict)
    if o is not None:
        if o.get("visibility") != "private":
            out.append(_f("PUBLIC_OUTPUT", "ERROR", "output.visibility", "must be 'private'; publishing is a separate approved step"))
        if o.get("push_to_hub") is not False:
            out.append(_f("AUTO_PUSH", "ERROR", "output.push_to_hub", "must be false; publish only after approval"))
        if o.get("overwrite_existing") is not False:
            out.append(_f("OVERWRITE", "ERROR", "output.overwrite_existing", "must be false (no model overwrite without approval)"))
        if not o.get("checkpoint_policy"):
            out.append(_f("MISSING", "WARN", "output.checkpoint_policy", "state optimizer/scheduler/RNG/data-loader resumability policy"))
    # contamination
    c = _req(m, "contamination", "", out, dict)
    if c is not None:
        if c.get("verdict") != "CLEAN_NOT_LAUNCHED":
            out.append(_f("CONTAMINATION_GATE", "ERROR", "contamination.verdict", "must be CLEAN_NOT_LAUNCHED from the gate report"))
        g = c.get("gate_report_sha256")
        if not (isinstance(g, str) and HEX64.match(g)):
            out.append(_f("BAD_SHA256", "ERROR", "contamination.gate_report_sha256", "64 hex of the gate report required"))
    # licensing
    lic = m.get("derived_license")
    if not lic:
        out.append(_f("MISSING", "ERROR", "derived_license", "declare the license the output will carry"))
    elif bm and bm.get("license_class") in RESTRICTED_BASE_CLASSES and str(lic).lower() in PERMISSIVE:
        out.append(_f("LICENSE_LAUNDERING", "ERROR", "derived_license",
                      f"base license_class={bm.get('license_class')} but derived_license={lic}; restricted-base derivatives cannot be relabelled permissive"))
    # secrets
    if SECRET_LIKE.search(json.dumps(m)):
        out.append(_f("SECRET_IN_MANIFEST", "ERROR", "*", "secret-looking value present (value not shown)"))
    errors = [x for x in out if x["severity"] == "ERROR"]
    return {"schema": "szl.train.run-manifest.validation/v1",
            "verdict": "INVALID" if errors else "VALID_NOT_LAUNCHED",
            "errors": len(errors), "warnings": len(out) - len(errors), "findings": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest")
    a = ap.parse_args(argv)
    res = validate(json.load(open(a.manifest)))
    print(json.dumps(res, indent=2))
    return 0 if res["verdict"] == "VALID_NOT_LAUNCHED" else 1


if __name__ == "__main__":
    sys.exit(main())
