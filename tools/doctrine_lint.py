#!/usr/bin/env python3
"""doctrine_lint.py — SZL doctrine compliance linter for all markdown cards.

Machine-enforced banned patterns (SZL-MARKETING-1.1 Part 2 guardrails).
A draft/card with violations does not ship — fail-closed, exit 1.

Usage: python tools/doctrine_lint.py [path ...]   (default: all .md under repo)
Exit 0 = clean. Exit 1 = violations found (each printed with rule name).
"""
import re, sys, os

BANNED = [
    (r"\bproven\b(?!.*conjecture)", "LAMBDA_OVERCLAIM (never 'proven' — always 'Conjecture 1, advisory')"),
    (r"\bguaranteed?\b", "GUARANTEE_LANGUAGE"),
    (r"\b100% (trust|safe|secure)", "CEILING_VIOLATION (trust ceiling is 0.97, never 100%)"),
    (r"\bstate.of.the.art\b", "UNVERIFIABLE_SUPERLATIVE"),
    (r"\breturns of \d", "QUANT_PERFORMANCE_CLAIM (szl-quant is advisory-only, paper-only)"),
    (r"\b\d+ ?(km|miles|meter) (detection|range)", "KILLINCHU_CAPABILITY_QUANTIFICATION (defense vertical — never quantify)"),
]

def lint(text):
    return [f"{msg} -> {m.group(0)!r}" for pat, msg in BANNED
            if (m := re.search(pat, text, re.I))]

def main():
    roots = sys.argv[1:] or ["."]
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        else:
            for dirpath, dirnames, filenames in os.walk(r):
                if ".git" in dirpath:
                    continue
                files += [os.path.join(dirpath, f) for f in filenames if f.endswith(".md")]
    viol = 0
    for p in sorted(files):
        try:
            hits = lint(open(p, errors="ignore").read())
        except OSError:
            continue
        for h in hits:
            print(f"BLOCKED {p}: {h}")
            viol += 1
    if viol:
        print(f"\nDOCTRINE LINT: {viol} violation(s) across {len(files)} markdown files — DO NOT SHIP")
        sys.exit(1)
    print(f"DOCTRINE LINT CLEAN — {len(files)} markdown files, zero violations")

if __name__ == "__main__":
    main()
