#!/usr/bin/env python3
"""Fail-closed Lean source + kernel axiom gate (never weakens; report-first).

Usage: python3 lean_axiom_gate.py PROJECT_DIR [--no-lake] [--exclude SUBSTR ...]
(--no-lake: standalone core-Lean files with no lakefile; uses `lean` directly.)

1. Source scan (a Lean-aware tokenizer drops comments and literal text and
   fails closed on anything it cannot lex unambiguously): any `sorry`, `admit`,
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


class StripError(Exception):
    """The source cannot be tokenized unambiguously; the gate fails closed."""


# Character classes follow Lean 4's own lexer (Lean.Parser.Basic):
#   isIdFirst c = c.isAlpha || c = '_' || isLetterLike c
#   isIdRest  c = c.isAlphanum || c = '_' || c = '\'' || c = '!' || c = '?'
#                 || isLetterLike c || isSubScriptAlnum c
def _letter_like(c):
    o = ord(c)
    return ((0x3B1 <= o <= 0x3C9 and o != 0x3BB)      # Greek lower, minus lambda
            or (0x391 <= o <= 0x3A9 and o not in (0x3A0, 0x3A3))
            or 0x3CA <= o <= 0x3FB or 0x1F00 <= o <= 0x1FFE
            or 0x2100 <= o <= 0x214F or 0x1D49C <= o <= 0x1D59F
            or (0x00C0 <= o <= 0x00FF and o not in (0xD7, 0xF7))   # Latin-1 letters
            or 0x0100 <= o <= 0x017F)                               # Latin Extended-A


def _subscript(c):
    o = ord(c)
    return (0x2080 <= o <= 0x2089 or 0x2090 <= o <= 0x209C or 0x1D62 <= o <= 0x1D6A
            or o == 0x2C7C)


def _id_first(c):
    return (c.isascii() and c.isalpha()) or c == "_" or _letter_like(c)


def _id_rest(c):
    return ((c.isascii() and c.isalnum()) or c in "_'!?"
            or _letter_like(c) or _subscript(c))


_CHAR = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|[\\\"'nrt])|[^\\'\n])'")
_INTERP_PREFIXES = ("s!", "m!", "f!")


def strip(s):
    """Return the code of a Lean source with comments removed and the text of
    every string/char literal replaced by an empty literal, keeping the code
    inside s!/m!/f! interpolation holes.

    This is a single-pass tokenizer (identifiers are lexed whole, as Lean
    does), not a set of look-behind heuristics. Anything it cannot lex
    unambiguously raises StripError, which the gate reports as a failure:
    unterminated comments, strings, raw strings, holes or «» names, a quote
    character that does not start a valid char literal at a token boundary,
    or an interpolation prefix that is not followed by a string."""
    out = []
    i = _lex(s, 0, out, in_hole=False)
    assert i == len(s)
    return "".join(out)


def _skip_ws_comments(s, i):
    """Index of the next non-whitespace, non-comment character (comments are
    whitespace in Lean). Raises on an unterminated block comment."""
    n = len(s)
    while i < n:
        if s[i] in " \t\r\n":
            i += 1
        elif s.startswith("--", i):
            while i < n and s[i] != "\n":
                i += 1
        elif s.startswith("/-", i):
            i = _block_comment_end(s, i)
        else:
            break
    return i


def _block_comment_end(s, i):
    depth, i, n = 1, i + 2, len(s)
    while i < n:
        if s.startswith("/-", i):
            depth += 1; i += 2
        elif s.startswith("-/", i):
            depth -= 1; i += 2
            if depth == 0:
                return i
        else:
            i += 1
    raise StripError("unterminated block comment")


def _lex(s, i, out, in_hole):
    """Lex code from i. In a hole, stop after the matching '}' and return the
    index after it; at top level, run to the end of input."""
    n, braces = len(s), 0
    while i < n:
        c = s[i]
        if s.startswith("--", i):                          # line comment; keep the newline
            while i < n and s[i] != "\n":
                i += 1
            continue
        if s.startswith("/-", i):
            i = _block_comment_end(s, i)
            continue
        if c == '"':
            out.append('"'); i = _string(s, i + 1, out, interp=False); continue
        if c == "'":
            m = _CHAR.match(s, i)
            if m:
                out.append("''"); i = m.end(); continue
            if s.startswith("''", i):                       # image notation f '' S
                out.append("''"); i += 2; continue
            if i and s[i - 1] in ")]":                      # xs[i]'h, (..)'h
                out.append("'"); i += 1; continue
            raise StripError(f"ambiguous quote at offset {i} (not a char literal)")
        if c == "\u00ab":                                   # «escaped name»
            j = s.find("\u00bb", i + 1)
            if j < 0:
                raise StripError("unterminated \u00ab\u00bb name")
            out.append(s[i:j + 1]); i = j + 1; continue
        if _id_first(c):
            j = i + 1
            while j < n and _id_rest(s[j]):
                j += 1
            word = s[i:j]
            k = j
            while k < n and s[k] == "#":
                k += 1
            if word == "r" and k < n and s[k] == '"':      # raw string r"..", r#".."#
                close = '"' + "#" * (k - j)
                e = s.find(close, k + 1)
                if e < 0:
                    raise StripError(f"unterminated raw string at offset {i}")
                out.append('""'); i = e + len(close); continue
            out.append(word); i = j
            if word in _INTERP_PREFIXES:
                k = _skip_ws_comments(s, i)
                if k >= n or s[k] != '"':
                    raise StripError(f"{word} at offset {i - len(word)} is not followed by a string")
                out.append(' "'); i = _string(s, k + 1, out, interp=True)
            continue
        if in_hole:
            if c == "{":
                braces += 1
            elif c == "}":
                if braces == 0:
                    out.append("}")
                    return i + 1
                braces -= 1
        out.append(c); i += 1
    if in_hole:
        raise StripError("unterminated interpolation hole")
    return i


def _string(s, i, out, interp):
    """Skip a string body from i (just after the opening quote); return the
    index after the closing quote. In an interpolated string, each {..} hole
    is lexed as code and kept."""
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            if i + 1 >= n:
                break
            i += 2; continue
        if c == '"':
            out.append('"'); return i + 1
        if interp and c == "{":
            out.append("{"); i = _lex(s, i + 1, out, in_hole=True); continue
        i += 1
    raise StripError("unterminated string literal")

# Every bypass reported against earlier versions of this scanner, plus basic
# shapes. main() runs these before scanning, so a regression in the lexer
# fails the gate instead of silently under-counting.
# `axiom` is a keyword, so after stripping comments and literals any standalone
# occurrence is an axiom command, whatever modifiers or attributes precede it
# (@[simp] axiom, noncomputable axiom, unsafe axiom, private axiom ...).
AXIOM_TOKEN = r"(?<![\w.'\u00ab])axiom(?![\w'!?\u00bb])"

SELF_TESTS = [
    ("theorem t : True := sorry", 1),
    ("def c := \'\"\'\ntheorem t : True := sorry", 1),          # '"' char literal
    ("def c := \'\\\'\'\ntheorem t : True := sorry", 1),      # '\'' char literal
    ('def c := s!"x {sorry} y"', 1),                            # interpolation hole
    ('def c := s!"x {f {a := 1}} {sorry}"\ntheorem t := sorry', 2),
    ('def c := s! "x {sorry}"', 1),                             # whitespace after s!
    ('def c := m!\n  "x {sorry}"', 1),
    ('def c := s! /- note -/ "x {sorry}"', 1),                  # comment after s!
    ('def c := s! -- note\n  "x {sorry}"', 1),
    ('-- s!\ndef c := "{" ++ sorry ++ "}"', 1),                 # s! inside a comment
    ('/- m! -/ "{" ++ sorry ++ "}"', 1),
    ('def c := f \'a\'r"x \\"\ntheorem t := sorry', 1),       # raw string after char lit
    ('def c := r#"a " sorry"#\ntheorem t := sorry', 1),
    ('def c := m!"{s!"in {sorry}"}"\ntheorem t := sorry', 2),
    ('def c := m!"{"inner {sorry}"}"\ntheorem t := sorry', 1),
    ('def c := "sorry"', 0),
    ("-- sorry\n/- sorry /- nested -/ sorry -/ theorem t := sorry", 1),
    ("theorem t (x' : Nat) : x' = x' := sorry", 1),
    ("example : (g '' S) = T := sorry", 1),
    ("example : xs[i]'h = 0 := sorry", 1),
    ("def caf\u00e9' := 1\ntheorem G\u00f6del' : True := sorry", 1),   # Latin-1 identifiers
    ("def \u0101' := r\ntheorem t := sorry", 1),                  # Latin Extended-A
    ("def x := r#[1, 2]\ntheorem t := sorry", 1),                  # r applied to #[..]
    ("def x := r ##[1]\ntheorem t := sorry", 1),
    ("@[simp] axiom a : False", 1),                                 # modifiers on axiom
    ("noncomputable axiom a : Nat", 1),
    ("theorem t : True := trivial unsafe axiom a : False", 1),
    ("private axiom a : False", 1),
    ("#print axioms t\ndef \u00abaxiom\u00bb := 1", 0),
]
SELF_TESTS_MUST_FAIL = [
    'def c := "abc',                     # unterminated string
    'def c := s!"x {sorry',              # unterminated hole
    "/- open",                           # unterminated comment
    'def c := r#"x',                     # unterminated raw string
    "def c := x + 'ab' + sorry",         # quote that is not a char literal
    "def c := s! 3",                     # prefix not followed by a string
]


def self_test():
    for src, want in SELF_TESTS:
        code = strip(src)
        got = len(re.findall(r"\bsorry\b", code)) + len(re.findall(AXIOM_TOKEN, code))
        if got != want:
            raise SystemExit(f"GATE FAILED: scanner self-test expected {want} sorry, got {got}: {src!r}")
    for src in SELF_TESTS_MUST_FAIL:
        try:
            strip(src)
        except StripError:
            continue
        raise SystemExit(f"GATE FAILED: scanner self-test accepted ambiguous source: {src!r}")


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
    self_test()
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
                        ("axiom", AXIOM_TOKEN)):
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
