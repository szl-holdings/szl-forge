#!/usr/bin/env python3
"""Fail-closed Lean source + kernel axiom gate (never weakens; report-first).

Usage: python3 lean_axiom_gate.py PROJECT_DIR [--no-lake] [--exclude SUBSTR ...]
(--no-lake: standalone core-Lean files with no lakefile; uses `lean` directly.)

1. Source scan (comments and string literals stripped): any `sorry`, `admit`,
   or `axiom` declaration in a tracked *.lean file under PROJECT_DIR fails.
2. Kernel scan: every `theorem`/`lemma` found is given `#print axioms`, the file
   is elaborated by `lake env lean`, and the gate fails if (a) the number of
   reported theorems differs from the number declared, or (b) any theorem
   depends on `sorryAx` or an axiom outside {propext, Classical.choice,
   Quot.sound}.
It prints the exact theorem count and per-axiom histogram so the numbers in a
README can be checked against CI output.
"""
import os, re, subprocess, sys

ALLOWED = {"propext", "Classical.choice", "Quot.sound"}


_CHAR = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'")
_IDENT = re.compile(r"[A-Za-z0-9_.!?'\u00C0-\uFFFF]")


class StripError(Exception):
    pass


def strip(s):
    """Drop comments and literal text, keep code (including `{...}` holes of
    interpolated strings such as s!"..."). Raises StripError on any literal or
    comment that does not terminate, so a malformed or adversarial file can
    never hide the rest of itself from the scan."""
    out, i, n = [], 0, len(s)
    depth = 0            # nested /- -/ comment depth
    holes = []           # brace depth stack for open interpolation holes
    lit_end = -1         # index just past the last char literal (a token boundary)
    while i < n:
        if depth:
            if s.startswith("/-", i):
                depth += 1; i += 2
            elif s.startswith("-/", i):
                depth -= 1; i += 2
            else:
                i += 1
            continue
        if s.startswith("/-", i):
            depth += 1; i += 2; continue
        if s.startswith("--", i):
            while i < n and s[i] != "\n":
                i += 1
            continue
        c = s[i]
        prev = s[i - 1] if i and i != lit_end else ""
        if holes:
            if c == "{":
                holes[-1] += 1
            elif c == "}":
                if holes[-1] == 0:
                    holes.pop()
                    out.append("}")
                    i = _string_body(s, i + 1, True, out)
                    if i < 0:
                        holes.append(0); i = -i
                    continue
                holes[-1] -= 1
        if c == "'" and not _IDENT.match(prev):
            m = _CHAR.match(s, i)
            if m:
                out.append("''"); i = lit_end = m.end(); continue
        if c == "r" and not _IDENT.match(prev):
            m = re.compile(r'r(#*)"').match(s, i)
            if m:
                close = '"' + m.group(1)
                j = s.find(close, m.end())
                if j < 0:
                    raise StripError(f"unterminated raw string at offset {i}")
                out.append('""'); i = j + len(close); continue
        if c == '"':
            interp = _interp_prefix(out)
            out.append('"')
            i = _string_body(s, i + 1, interp, out)
            if i < 0:
                holes.append(0); i = -i
            continue
        out.append(c); i += 1
    if depth:
        raise StripError("unterminated block comment")
    if holes:
        raise StripError("unterminated interpolated string")
    return "".join(out)


def _interp_prefix(out):
    """True when the quote about to be emitted opens an s!/m!/f! interpolated
    string. Looks back over the already-stripped code (comments removed, so a
    prefix inside a comment never counts, and a comment between the prefix
    and the quote is just whitespace, as in Lean)."""
    tail = []
    for piece in reversed(out):
        tail.append(piece)
        if sum(len(p) for p in tail) >= 64:
            break
    t = "".join(reversed(tail)).rstrip(" \t\r\n")
    if len(t) < 2 or t[-1] != "!" or t[-2] not in "smf":
        return False
    return len(t) < 3 or not _IDENT.match(t[-3])


def _string_body(s, i, interp, out):
    """Skip a string body starting at i. Returns the index after the closing
    quote, or -(index after '{') when an interpolation hole opens."""
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2; continue
        if c == '"':
            out.append('"'); return i + 1
        if interp and c == "{":
            out.append("{"); return -(i + 1)
        i += 1
    raise StripError("unterminated string literal")


DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|nonrec\s+)*(?:theorem|lemma)\s+([^\s:({\[]+)")
NS = re.compile(r"^\s*namespace\s+(\S+)")
END = re.compile(r"^\s*end(?:\s+(\S+))?\s*$")
SEC = re.compile(r"^\s*section\b")


def theorems(code):
    stack, names = [], []
    for line in code.split("\n"):
        m = NS.match(line)
        if m:
            stack.append(("ns", m.group(1))); continue
        if SEC.match(line):
            stack.append(("sec", None)); continue
        m = END.match(line)
        if m and stack:
            stack.pop(); continue
        m = DECL.match(line)
        if m:
            nm = m.group(1)
            ns = ".".join(x for k, x in stack if k == "ns")
            names.append((ns + "." if ns else "") + nm)
    return names


def main():
    args = sys.argv[1:]
    proj = args[0]
    cmd = ["lean"] if "--no-lake" in args else ["lake", "env", "lean"]
    excl = [args[i + 1] for i, a in enumerate(args) if a == "--exclude"]
    files = []
    for root, dirs, fs in os.walk(proj):
        dirs[:] = [d for d in dirs if d not in (".lake", "build", ".git")]
        for f in fs:
            p = os.path.join(root, f)
            if f.endswith(".lean") and f != "lakefile.lean" and not any(e in p for e in excl):
                files.append(p)
    files.sort()
    bad, total, hist = [], 0, {}
    for p in files:
        raw = open(p, encoding="utf-8", errors="replace").read()
        try:
            code = strip(raw)
        except StripError as e:
            bad.append(f"{p}: cannot scan source safely: {e}")
            continue
        for tok, rx in (("sorry", r"\bsorry\b"), ("admit", r"\badmit\b"),
                        ("axiom", r"(?m)^\s*(?:private\s+|protected\s+)?axiom\b")):
            c = len(re.findall(rx, code))
            if c:
                bad.append(f"{p}: {c} x {tok}")
        names = theorems(code)
        if not names:
            continue
        total += len(names)
        # kernel scan: append #print axioms to a scratch copy next to the original
        scratch = p[:-5] + "__axiomgate.lean"
        with open(scratch, "w", encoding="utf-8") as fh:
            fh.write(raw + "\n" + "\n".join(f"#print axioms {n}" for n in names) + "\n")
        try:
            r = subprocess.run(cmd + [scratch], cwd=proj,
                               capture_output=True, text=True)
        finally:
            os.remove(scratch)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            bad.append(f"{p}: lean elaboration failed:\n{out[-2000:]}")
            continue
        seen = set()
        for line in out.split("\n"):
            m = re.match(r"'(.+)' depends on axioms: \[(.*)\]", line)
            if m:
                if m.group(1) in seen:
                    continue  # the source may already #print this theorem
                seen.add(m.group(1))
                for a in [x.strip() for x in m.group(2).split(",") if x.strip()]:
                    hist[a] = hist.get(a, 0) + 1
                    if a not in ALLOWED:
                        bad.append(f"{p}: {m.group(1)} depends on {a}")
            else:
                m = re.match(r"'(.+)' does not depend on any axioms", line)
                if m:
                    seen.add(m.group(1))
        missing = sorted(set(names) - seen)
        if missing:
            bad.append(f"{p}: kernel gave no axiom report for {missing}")
    print(f"files_scanned={len(files)} theorems={total} axiom_histogram={hist}")
    if bad:
        print("GATE FAILED:")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("GATE OK: no sorry/admit/axiom declarations; every declared theorem was kernel-reported and uses only", sorted(ALLOWED))


main()
