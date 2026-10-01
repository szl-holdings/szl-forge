#!/usr/bin/env python3
# =============================================================================
# szl_geh_v8.py — SZL Governed Evaluation Harness v8 (Lean 4 lane)
#
# What this is: a fail-closed harness that turns LLM/agent theorem-proving into
# governed, hash-linked tool-calls against a REAL Lean 4 kernel (Lean REPL,
# leanprover/lean4:v4.18.0 — the lutar-lean toolchain), audits the finished
# proof's axioms through a build-failing guard (#geh_guard), and signs the
# result as a DSSE proof receipt in an append-only chain.
#
# What changed vs v3..v7 (all of which were never executed): see the audit
# table in GEH_v8_CODEX_PAYLOAD.md. Every function here has run in CI-style
# tests (tests/test_geh_v8.py) and the Lean files compiled under v4.18.0.
#
# Doctrine: fail-closed, local-only. No network, training, Hub or Git writes.
#   1. sorry/admit/native_decide/foreign axioms VOID a receipt — always,
#      enforced twice: at the tactic gate AND by the kernel-side axiom audit.
#   2. plausible/slim_check are EVIDENCE_ONLY; they can never satisfy a gate.
#   3. SIMULATED transport can never produce a PASS-grade receipt.
#   4. A PASS requires: kernel closed all goals (live channel) AND the
#      reconstructed file compiles with zero errors/sorries (audit channel)
#      AND #geh_guard reports allowed=true. Three independent checks.
#
# CLI (all local):
#   --selfcheck                      structural + (if REPL present) kernel tests
#   --emit-lean [--lean-project P]   write the geh-lean lake project
#   --gen-crypto K                   write a randomized crypto-property suite
#   --prove "STMT" [--transport ...] governed proof loop -> signed receipt
#   --bench-crypto K [--transport …] contamination-resistant prover benchmark
#   --audit-file F.lean              audit channel only (compile + guard)
#   --verify FILE | --verify-chain   DSSE + chain verification
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

HV = "szl-geh/8.0.0"
PT = "application/vnd.szl.gov.proof-receipt+json"
RECEIPT_TYPE = "szl.gov.proof.receipt/v2"
TOOLSPEC_ID = "szl.gov.lean.tools/v2"
LEAN_TOOLCHAIN = "leanprover/lean4:v4.18.0"      # lutar-lean pin
REPL_REV = "v4.18.0"                              # leanprover-community/repl tag
ALLOWED_AXIOMS = ("Classical.choice", "Quot.sound", "propext")

# ----------------------------------------------------------------------------
# Roots. Windows owner-metal default matches the thread; Linux/CI falls back
# to the current directory. Override with --root / SZL_FORGE_ROOT.
# ----------------------------------------------------------------------------
def default_root() -> Path:
    env = os.environ.get("SZL_FORGE_ROOT")
    if env:
        return Path(env)
    win = Path(r"C:\Users\steph\szl-forge")
    return win if win.exists() else Path.cwd()


@dataclass
class Paths:
    root: Path
    lean_project: Path
    evidence: Path = field(init=False)
    chain: Path = field(init=False)
    a12log: Path = field(init=False)
    keydir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.evidence = self.root / "chaski_r4" / "evidence" / "geh"
        self.chain = self.evidence / "geh_chain.jsonl"
        self.a12log = self.evidence / "geh_article12_log.jsonl"
        self.keydir = self.evidence / "geh_keys"
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.keydir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Canonical JSON + hashing
# ----------------------------------------------------------------------------
def canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(o: Any) -> str:
    return hashlib.sha256(canon(o).encode("utf-8")).hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canary(run_id: str, tag: str, i: int) -> str:
    return f"SZL-CANARY-{run_id[:8]}-{tag}-{i:03d}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------
# DSSE (ed25519 via `cryptography`; honest UNSIGNED fallback that can never
# verify). PAE per https://github.com/secure-systems-lab/dsse/blob/master/protocol.md
# ----------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    HAS_ED = True
except Exception:  # pragma: no cover
    HAS_ED = False


def pae(payload_type: str, payload: bytes) -> bytes:
    return (
        b"DSSEv1 "
        + str(len(payload_type.encode("utf-8"))).encode()
        + b" "
        + payload_type.encode("utf-8")
        + b" "
        + str(len(payload)).encode()
        + b" "
        + payload
    )


class DSSESigner:
    def __init__(self, keydir: Path):
        self.kind, self._k, self.keyid, self.pub_raw = "UNSIGNED_HONEST", None, "none", None
        if not HAS_ED:
            return
        pem = keydir / "geh_ed25519.pem"
        pub = keydir / "geh_ed25519.pub"
        if pem.exists():
            self._k = serialization.load_pem_private_key(pem.read_bytes(), password=None)
        else:
            self._k = Ed25519PrivateKey.generate()
            pem.write_bytes(
                self._k.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
        self.pub_raw = self._k.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        if not pub.exists():
            pub.write_bytes(
                self._k.public_key().public_bytes(
                    serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
                )
            )
        # keyid = sha256 of the raw 32-byte public key (stable, portable)
        self.keyid = sha_bytes(self.pub_raw)[:16]
        self.kind = "ed25519"
        (keydir / "geh_ed25519.pub.hex").write_text(self.pub_raw.hex() + "\n", encoding="utf-8")

    def sign(self, payload: dict) -> dict:
        body = canon(payload).encode("utf-8")
        if self.kind == "ed25519":
            sig = self._k.sign(pae(PT, body)).hex()
            return {
                "payloadType": PT,
                "payload": body.decode("utf-8"),
                "signatures": [{"keyid": self.keyid, "sig": sig, "alg": "ed25519", "pub_hex": self.pub_raw.hex()}],
            }
        return {
            "payloadType": PT,
            "payload": body.decode("utf-8"),
            "signatures": [{"keyid": "none", "sig": "UNSIGNED_HONEST", "alg": "none"}],
        }

    @staticmethod
    def verify(env: dict, pub: "Ed25519PublicKey") -> bool:
        try:
            s = env["signatures"][0]
            if s.get("alg") != "ed25519":
                return False
            pub.verify(bytes.fromhex(s["sig"]), pae(env["payloadType"], env["payload"].encode("utf-8")))
            return True
        except Exception:
            return False


def load_pub(keydir: Path):
    return serialization.load_pem_public_key((keydir / "geh_ed25519.pub").read_bytes())


# ----------------------------------------------------------------------------
# Tactic gate registry (the governed part of every kernel call)
# ----------------------------------------------------------------------------
TACTIC_REGISTRY: dict[str, str] = {
    # SAFE: stateless / state-advancing / kernel-checkable
    "intro": "SAFE", "intros": "SAFE", "exact": "SAFE", "apply": "SAFE", "rfl": "SAFE",
    "decide": "SAFE", "simp": "SAFE", "simp_all": "SAFE", "omega": "SAFE", "constructor": "SAFE",
    "cases": "SAFE", "rcases": "SAFE", "rw": "SAFE", "rewrite": "SAFE", "exact?": "BLOCKED",
    "trivial": "SAFE", "assumption": "SAFE", "contradiction": "SAFE", "left": "SAFE",
    "right": "SAFE", "exists": "SAFE", "use": "SAFE", "rotate_left": "SAFE",
    "rotate_right": "SAFE", "case": "SAFE", "next": "SAFE", "all_goals": "SAFE",
    "any_goals": "SAFE", "first": "SAFE", "try": "SAFE", "ring": "SAFE", "norm_num": "SAFE",
    # RESTRICTED: allowed, but recorded with a continuity hash and counted
    "induction": "RESTRICTED", "refine": "RESTRICTED", "have": "RESTRICTED",
    "let": "RESTRICTED", "show": "RESTRICTED", "conv": "RESTRICTED", "calc": "RESTRICTED",
    "focus": "RESTRICTED", "obtain": "RESTRICTED", "generalize": "RESTRICTED",
    "unfold": "RESTRICTED", "change": "RESTRICTED", "subst": "RESTRICTED",
    # EVIDENCE_ONLY: property-based testing — never satisfies a compliance gate
    "plausible": "EVIDENCE_ONLY", "slim_check": "EVIDENCE_ONLY",
    # BLOCKED: fail-closed. native_decide adds Lean.ofReduceBool (foreign axiom).
    "sorry": "BLOCKED", "admit": "BLOCKED", "axiom": "BLOCKED", "native_decide": "BLOCKED",
    "decide!": "BLOCKED", "exfalso": "SAFE",
}
GATE_OK = ("OK", "OK_RESTRICTED")

_HEAD_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_'!?]*)")


def tactic_head(tactic: str) -> str:
    m = _HEAD_RE.match(tactic or "")
    return m.group(1) if m else ""


def gate_tactic(tactic: str) -> str:
    """Return OK | OK_RESTRICTED | EVIDENCE_ONLY | BLOCKED | UNREGISTERED.
    Combinators are scanned too: `cases a <;> sorry` is BLOCKED."""
    if not tactic or not tactic.strip():
        return "UNREGISTERED"
    # every token that looks like a tactic head inside combinators
    heads = [tactic_head(part) for part in re.split(r"<;>|;|\n|\(|\)", tactic) if part.strip()]
    worst = "OK"
    order = {"OK": 0, "OK_RESTRICTED": 1, "UNREGISTERED": 2, "EVIDENCE_ONLY": 3, "BLOCKED": 4}
    for i, h in enumerate(heads):
        if not h:
            continue
        risk = TACTIC_REGISTRY.get(h)
        if risk is None:
            # words after the first head are usually terms (e.g. `exact h`), not tactics
            verdict = "UNREGISTERED" if i == 0 else "OK"
        else:
            verdict = {"SAFE": "OK", "RESTRICTED": "OK_RESTRICTED"}.get(risk, risk)
        if order[verdict] > order[worst]:
            worst = verdict
    return worst


# ----------------------------------------------------------------------------
# Artifact classifier (Plausible isolation, layer 2)
# ----------------------------------------------------------------------------
EVIDENCE_MARKERS = ("import Plausible", "by plausible", "SlimCheck", "slim_check", "#eval ", "GEH-EVIDENCE-ONLY")
QUARANTINE_RE = re.compile(r"(?<![A-Za-z0-9_'])(sorry|admit|native_decide|decide!)(?![A-Za-z0-9_'])")
AXIOM_DECL_RE = re.compile(r"^\s*axiom\s", re.MULTILINE)


def classify_lean_artifact(text: str) -> str:
    """QUARANTINED | EVIDENCE_ONLY | COMPLIANCE_CANDIDATE (still needs the kernel + guard)."""
    stripped = strip_lean_comments(text)
    if QUARANTINE_RE.search(stripped) or AXIOM_DECL_RE.search(stripped):
        return "QUARANTINED"
    if any(m in stripped for m in EVIDENCE_MARKERS):
        return "EVIDENCE_ONLY"
    return "COMPLIANCE_CANDIDATE"


def strip_lean_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", text)


# ----------------------------------------------------------------------------
# Tool-use spec (state + tools + machine). This is the contract an agent sees.
# ----------------------------------------------------------------------------
TOOLSPEC: dict = {
    "$id": TOOLSPEC_ID,
    "state": {
        "state_id": "string (transport-native id; REPL proofState / Pantograph stateId)",
        "goals": [{"goal_id": "int", "pp": "string", "pp_sha256": "string"}],
        "steps_used": "int",
        "budget": "int",
    },
    "tools": {
        "lean.initialize": {"args": {"statement": "string"}, "post": "state with >=0 goals; kernel elaborated the statement"},
        "lean.list_goals": {"args": {"state_id": "string"}, "post": "goals with per-goal sha256"},
        "lean.run_tactic": {
            "args": {"state_id": "string", "goal_id": "int", "tactic": "string"},
            "pre": "goal exists AND gate(tactic) in {OK, OK_RESTRICTED} AND budget > 0",
            "post": "ok AND new goals (possibly []) OR branch_death{KERNEL|BLOCKED|EVIDENCE_ONLY|UNREGISTERED|BUDGET}",
            "risk": "SAFE|RESTRICTED|EVIDENCE_ONLY|BLOCKED",
        },
        "lean.focus_goal": {"args": {"state_id": "string", "goal_id": "int"}, "post": "goal_id rotated to position 0 (recorded, governed)"},
        "lean.finalize": {"pre": "no goals remain", "post": "tactic script reconstructed for the audit channel"},
        "lean.audit_file": {"args": {"file": "string"}, "post": "kernel compiled the whole file: errors, sorries, allTactics extracted"},
        "lean.query_axioms": {"args": {"theorem": "string"}, "post": "axiom set subset of ALLOWED_AXIOMS else BUILD FAILS"},
    },
    "machine": "INIT -> GOALS -> (RUN_TACTIC -> GOALS)* -> FINALIZE -> AUDIT_FILE -> QUERY_AXIOMS -> RECEIPT; dead branches recorded, never silently retried",
    "verdicts": ["KERNEL_VERIFIED_AND_SIGNED", "TRACE_SIGNED_KERNEL_UNVERIFIED", "VOID"],
    "pass_requires": ["transport.certified", "live: no goals", "audit: 0 errors, 0 sorries", "guard: allowed=true", "artifact_grade=COMPLIANCE_CANDIDATE"],
}


# ----------------------------------------------------------------------------
# Transports. State shape is shared: {"state_id", "goals":[{"goal_id","pp"}],
# "ok": bool, "message": str|None, "raw": transport-native}
# ----------------------------------------------------------------------------
class Transport:
    name = "BASE"
    certified = False          # can this transport ever yield PASS?
    verified_here = False      # was this transport exercised against a kernel in this build?

    def start(self, statement: str, prelude: str = "", name: str = "geh_goal") -> dict:
        raise NotImplementedError

    def step(self, state: dict, goal_id: int, tactic: str) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SimulatedTransport(Transport):
    """Deterministic toy. NEVER certifies. Exists so the governance loop can be
    unit-tested without a kernel; a receipt from it is always KERNEL_UNVERIFIED."""

    name, certified, verified_here = "SIMULATED", False, True

    def start(self, statement, prelude="", name="geh_goal"):
        goals = [{"goal_id": 0, "pp": f"⊢ {statement}"}]   # like a kernel: one main goal
        return {"state_id": "S0", "goals": goals, "ok": True, "message": None, "steps": 0}

    def step(self, state, goal_id, tactic):
        head = tactic_head(tactic)
        rest = [g for g in state["goals"] if g["goal_id"] != goal_id]
        if head == "constructor":
            base = 10 * (state["steps"] + 1)
            rest = [{"goal_id": base, "pp": "⊢ left"}, {"goal_id": base + 1, "pp": "⊢ right"}] + rest
        elif head in ("intro", "intros"):
            rest = [{"goal_id": goal_id, "pp": "h : hyp\n⊢ goal'"}] + rest
        elif head in ("exact", "rfl", "decide", "assumption", "trivial", "omega", "simp"):
            pass  # closes the goal
        else:
            return {"state_id": state["state_id"], "goals": state["goals"], "ok": False,
                    "message": f"simulated kernel: unknown tactic {head}", "steps": state["steps"] + 1}
        return {"state_id": f"S{state['steps'] + 1}", "goals": rest, "ok": True, "message": None,
                "steps": state["steps"] + 1}


class LeanReplTransport(Transport):
    """Kernel path via leanprover-community/repl (JSON over stdin/stdout).
    `sorry` opens a proofState; `{"tactic", "proofState"}` steps it; goals come
    back as pretty-printed strings. Proof states are immutable, so retrying a
    different tactic from the same state is true backtracking.

    Resource discipline (learned the hard way: an unbounded REPL froze a 2-core
    box by accumulating environments): one base environment holds the imports
    and every command derives from it; each RPC has a wall-clock timeout and
    the child runs under an address-space limit on POSIX. A timeout is a dead
    branch (KERNEL_TIMEOUT) and the process is respawned — never a hang."""

    name, certified = "LEAN_REPL", True
    IMPORTS = "import GEH.Compliance.GehGuard"

    def __init__(self, lean_project: Path, timeout: float = 90.0, mem_mb: int = 3072):
        self.project = Path(lean_project)
        self.timeout, self.mem_mb = timeout, mem_mb
        self.proc: Optional[subprocess.Popen] = None
        self.base_env: Optional[int] = None
        self.rpc_count = 0
        self.respawns = 0
        self.verified_here = True
        self._spawn()

    # -- process management ---------------------------------------------------
    def _preexec(self):
        try:
            import resource
            lim = self.mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except Exception:
            pass

    def _spawn(self) -> None:
        import queue, threading
        exe = repl_binary(self.project)
        if exe is None:
            raise SystemExit(f"FAIL-CLOSED: REPL binary not built under {self.project}")
        # `lake env` sets LEAN_PATH so `import GEH.…` resolves inside the REPL
        cmd = ["lake", "env", str(exe)]
        kw = {}
        if os.name == "posix":
            kw["preexec_fn"] = self._preexec
        self.proc = subprocess.Popen(cmd, cwd=str(self.project), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1, **kw)
        self.base_env = None
        # cross-platform (Windows pipes are not selectable): a reader thread feeds a queue
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        proc, q = self.proc, self._q

        def _reader():
            try:
                for line in proc.stdout:
                    q.put(line)
            except Exception:
                pass
            q.put(None)

        self._reader = threading.Thread(target=_reader, daemon=True)
        self._reader.start()

    def _respawn(self) -> None:
        self.close()
        self.respawns += 1
        self._spawn()

    def _rpc(self, obj: dict) -> dict:
        import queue
        assert self.proc and self.proc.stdin
        self.rpc_count += 1
        try:
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._respawn()
            return {"error": "KERNEL_DIED"}
        buf, deadline = [], time.time() + self.timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self._respawn()
                return {"error": "KERNEL_TIMEOUT"}
            try:
                line = self._q.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if self.proc.poll() is not None:
                    self._respawn()
                    return {"error": "KERNEL_DIED"}
                continue
            if line is None:
                self._respawn()
                return {"error": "KERNEL_DIED"}
            if line.strip() == "" and buf:
                break
            if line.strip():
                buf.append(line)
        try:
            return json.loads("".join(buf))
        except json.JSONDecodeError:
            return {"error": "KERNEL_BAD_JSON", "raw": "".join(buf)[:500]}

    def close(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            self.base_env = None

    def _ensure_base(self) -> Optional[int]:
        if self.base_env is None:
            resp = self._rpc({"cmd": self.IMPORTS})
            if "env" in resp:
                self.base_env = resp["env"]
                self._prelude_envs = {}
        return self.base_env

    def _prelude_env(self, prelude: str) -> Optional[int]:
        """Definitions must live in their own environment BEFORE the theorem is
        opened: a `sorry` proofState does not see constants declared in the same
        command (REPL snapshot semantics — 'unknown constant' otherwise)."""
        base = self._ensure_base()
        if base is None:
            return None
        if not prelude.strip():
            return base
        key = sha(prelude)
        envs = getattr(self, "_prelude_envs", {})
        if key not in envs:
            resp = self._rpc({"cmd": prelude, "env": base})
            if "env" not in resp or self._errors(resp):
                return None
            envs[key] = resp["env"]
            self._prelude_envs = envs
        return envs[key]

    # -- transport API --------------------------------------------------------
    @staticmethod
    def _goals(resp: dict) -> list[dict]:
        return [{"goal_id": i, "pp": g} for i, g in enumerate(resp.get("goals", []))]

    @staticmethod
    def _errors(resp: dict) -> list[str]:
        errs = [m.get("data", "") for m in resp.get("messages", []) if m.get("severity") == "error"]
        if "error" in resp:
            errs.append(str(resp["error"]))
        if "message" in resp and "proofState" not in resp and "env" not in resp:
            errs.append(str(resp["message"]))
        return errs

    def start(self, statement, prelude="", name="geh_goal"):
        env = self._prelude_env(prelude)
        if env is None:
            return {"state_id": None, "goals": [], "ok": False, "message": "environment unavailable (imports/prelude failed)", "steps": 0}
        src = f"theorem {name} : {statement} := by\n  sorry\n"
        resp = self._rpc({"cmd": src, "env": env})
        errs = self._errors(resp)
        if errs or not resp.get("sorries"):
            return {"state_id": None, "goals": [], "ok": False, "message": "; ".join(errs)[:400] or "no sorry proofState",
                    "steps": 0, "env": resp.get("env")}
        s = resp["sorries"][0]
        return {"state_id": s["proofState"], "goals": [{"goal_id": 0, "pp": s["goal"]}], "ok": True, "message": None,
                "steps": 0, "env": resp.get("env"), "elaborated_goal_sha256": sha(s["goal"])}

    def step(self, state, goal_id, tactic):
        resp = self._rpc({"tactic": tactic, "proofState": state["state_id"]})
        errs = self._errors(resp)
        if resp.get("sorries"):
            errs.append("tactic introduced sorry")
        ok = not errs and "proofState" in resp
        return {"state_id": resp.get("proofState", state["state_id"]), "goals": self._goals(resp) if ok else state["goals"],
                "ok": ok, "message": "; ".join(errs)[:400] or None, "steps": state["steps"] + 1}

    # -- audit channel ----------------------------------------------------------
    def _guard_in_env(self, env: int, theorems: list[str]) -> list[dict]:
        out = []
        for thm in theorems:
            g = self._rpc({"cmd": f"#geh_guard {thm}", "env": env})
            line = next((m.get("data", "") for m in g.get("messages", []) if "GEH_GUARD" in m.get("data", "")), "")
            parsed = None
            if line:
                try:
                    parsed = json.loads(line.split("GEH_GUARD", 1)[1].strip().splitlines()[0])
                except Exception:
                    parsed = None
            gerr = self._errors(g)
            out.append({"theorem": thm, "allowed": bool(parsed and parsed.get("allowed") is True and not gerr),
                        "axioms": (parsed or {}).get("axioms"), "raw": line[:400], "errors": [e[:300] for e in gerr[:3]]})
        return out

    def _audit_resp(self, resp: dict, guard_theorems: list[str]) -> dict:
        errors = self._errors(resp)
        sorries = resp.get("sorries", [])
        tactics = resp.get("tactics", [])
        env = resp.get("env")
        guard = self._guard_in_env(env, guard_theorems) if (env is not None and not errors) else []
        invocations = [{"tactic": t.get("tactic"), "goals_before": t.get("goals"), "pos": t.get("pos"),
                        "triple_sha256": sha([t.get("goals"), t.get("tactic")])} for t in tactics]
        return {"errors": [e[:300] for e in errors], "error_count": len(errors), "sorry_count": len(sorries),
                "tactic_count": len(tactics), "invocations": invocations, "invocations_sha256": sha(invocations),
                "guard": guard, "guard_all_allowed": bool(guard) and all(g["allowed"] for g in guard)}

    def audit_source(self, src_without_imports: str, guard_theorems: list[str]) -> dict:
        """Compile reconstructed source in the base environment (no re-import),
        extracting every tactic invocation, then run #geh_guard per theorem."""
        base = self._ensure_base()
        if base is None:
            return {"errors": ["base environment unavailable"], "error_count": 1, "sorry_count": 0, "tactic_count": 0,
                    "invocations": [], "invocations_sha256": sha([]), "guard": [], "guard_all_allowed": False}
        body = "\n".join(l for l in src_without_imports.splitlines() if not l.startswith("import "))
        resp = self._rpc({"cmd": body, "env": base, "allTactics": True})
        return self._audit_resp(resp, guard_theorems)

    def audit_file(self, path: Path, guard_theorems: list[str]) -> dict:
        """Compile a whole file (with its own imports) through the kernel with
        allTactics extraction, then run #geh_guard for each theorem."""
        resp = self._rpc({"path": str(path), "allTactics": True})
        return self._audit_resp(resp, guard_theorems)


class PantographTransport(Transport):
    """Kernel path via PyPantograph (stanford-centaur). API pinned to the
    current server.py: Server(imports, project_path); goal_start(expr);
    goal_tactic(state, tactic, site) — goal selection is a Site, NOT a goal_id
    kwarg (v6/v7 had this wrong). Marked verified_here=False unless the package
    imported and a probe proof closed in this process."""

    name, certified = "PANTOGRAPH", True

    def __init__(self, project_path: Optional[Path] = None, imports=("Init",)):
        from pantograph import Server  # type: ignore
        try:
            from pantograph.expr import Site  # type: ignore
        except Exception:  # pragma: no cover
            Site = None
        self._Site = Site
        kw = {"imports": list(imports)}
        if project_path:
            kw["project_path"] = str(project_path)
        self.server = Server(**kw)
        self.verified_here = False

    def start(self, statement, prelude="", name="geh_goal"):
        st = self.server.goal_start(statement)
        goals = [{"goal_id": i, "pp": str(g)} for i, g in enumerate(st.goals)]
        return {"state_id": st.state_id, "goals": goals, "ok": True, "message": None, "steps": 0, "raw": st}

    def step(self, state, goal_id, tactic):
        try:
            site = self._Site(goal_id=goal_id) if (self._Site and goal_id) else (self._Site() if self._Site else None)
            res = self.server.goal_tactic(state["raw"], tactic, site) if site is not None else self.server.goal_tactic(state["raw"], tactic)
            goals = [{"goal_id": i, "pp": str(g)} for i, g in enumerate(res.goals)]
            return {"state_id": res.state_id, "goals": goals, "ok": True, "message": None, "steps": state["steps"] + 1, "raw": res}
        except Exception as e:  # dead branch (metavariable loss, tactic error)
            return {"state_id": state["state_id"], "goals": state["goals"], "ok": False, "message": str(e)[:300],
                    "steps": state["steps"] + 1, "raw": state.get("raw")}


class LeanDojoTransport(Transport):
    """Gym loop adapter (Dojo(theorem) -> run_tac -> TacticState|ProofFinished).
    NOT exercised against a traced repo in this build (LeanDojo needs a traced
    LeanGitRepo). The wiring is unit-tested with a fake dojo; receipts record
    verified_here=False. Do not treat a LeanDojo receipt as PASS until this
    transport has been exercised on your metal."""

    name, certified = "LEANDOJO", True

    def __init__(self, dojo=None, init_state=None, proof_finished_cls=None):
        self.dojo, self.state0, self._PF = dojo, init_state, proof_finished_cls
        self.verified_here = False

    @staticmethod
    def _goals(pp: str) -> list[dict]:
        blocks = [b for b in re.split(r"\n\s*\n", pp or "") if "⊢" in b]
        return [{"goal_id": i, "pp": b} for i, b in enumerate(blocks)] if blocks else ([{"goal_id": 0, "pp": pp}] if pp else [])

    def start(self, statement, prelude="", name="geh_goal"):
        return {"state_id": str(getattr(self.state0, "id", 0)), "goals": self._goals(getattr(self.state0, "pp", "")),
                "ok": True, "message": None, "steps": 0, "raw": self.state0}

    def step(self, state, goal_id, tactic):
        res = self.dojo.run_tac(state["raw"], tactic)
        if self._PF is not None and isinstance(res, self._PF):
            return {"state_id": "finished", "goals": [], "ok": True, "message": None, "steps": state["steps"] + 1, "raw": res}
        msg = getattr(res, "message", None) or getattr(res, "error", None)
        ok = msg is None and hasattr(res, "pp")
        return {"state_id": str(getattr(res, "id", "?")), "goals": self._goals(getattr(res, "pp", "")) if ok else state["goals"],
                "ok": ok, "message": None if ok else str(msg or "dojo error")[:300], "steps": state["steps"] + 1, "raw": res}


# ----------------------------------------------------------------------------
# Provers / proposers. propose(goal_pp, history) -> list[str] candidates.
# ----------------------------------------------------------------------------
class ScriptedProver:
    """Deterministic per-goal scripts, advancing with the trajectory. NOT an LLM."""

    def __init__(self, script: list[str]):
        self.script, self.i = list(script), 0

    def propose(self, goal_pp: str, history: list) -> list[str]:
        if self.i >= len(self.script):
            return []
        t = self.script[self.i]
        self.i += 1
        return [t]


class HeuristicProver:
    """Closed-term prover for the crypto families: tries kernel-certain closers
    in a fixed order. Candidate lists give the frontier loop real backtracking."""

    def propose(self, goal_pp: str, history: list) -> list[str]:
        target = goal_pp.split("⊢", 1)[-1].strip()
        m = re.match(r"∀\s*\(([^:()]+):", target)
        if m:                                   # introduce exactly the leading binders
            return [f"intro {' '.join(m.group(1).split())}", "intros"]
        bools = re.findall(r"^([A-Za-z_][A-Za-z0-9_']*(?: [A-Za-z_][A-Za-z0-9_']*)*) : Bool$", goal_pp, flags=re.MULTILINE)
        if bools and ("Bool.xor" in target or "^^" in target or " xor " in target):
            names = " ".join(bools).split()
            return [" <;> ".join(f"cases {n}" for n in names) + " <;> rfl", "decide"]
        if "≠" in target:
            return ["decide", "simp"]
        return ["rfl", "decide", "simp", "omega"]


class CallableProposer:
    """Real LLM hook: fn(goal_pp, history) -> str | list[str]."""

    def __init__(self, fn: Callable[[str, list], Any]):
        self.fn = fn

    def propose(self, goal_pp: str, history: list) -> list[str]:
        out = self.fn(goal_pp, list(history))
        return [out] if isinstance(out, str) else list(out or [])


# ----------------------------------------------------------------------------
# Governed proof search: goal-addressed, candidate-backtracking, budgeted.
# Every kernel call is preceded by the gate and followed by a hash-linked event.
# ----------------------------------------------------------------------------
def prove(transport: Transport, statement: str, prover, *, prelude: str = "", name: str = "geh_goal",
          max_steps: int = 48, max_depth: int = 24) -> dict:
    events: list[dict] = [{"tool": "lean.initialize", "statement": statement, "statement_sha256": sha(statement),
                           "transport": transport.name, "ts": utcnow()}]
    st = transport.start(statement, prelude=prelude, name=name)
    events[0].update(ok=st["ok"], message=st.get("message"), elaborated_goal_sha256=st.get("elaborated_goal_sha256"))
    tree = {"node": "root", "children": [], "verdict": "OPEN"}
    if not st["ok"]:
        tree["verdict"] = "DEAD_INIT"
        events.append({"tool": "lean.finalize", "no_goals": False, "reason": "INIT_FAILED"})
        return {"events": events, "tree": tree, "done": False, "script": [], "stats": {"steps": 0, "dead": 1, "depth": 0}}

    steps = dead = 0
    history: list[tuple[str, str]] = []
    script: list[str] = []          # successful tactic path (for the audit file)
    node = tree
    depth = 0
    done = False
    while steps < max_steps and depth < max_depth:
        if not st["goals"]:
            done = True
            node["verdict"] = "CLOSED"
            break
        g = st["goals"][0]           # goal 0 is the kernel's main goal
        candidates = prover.propose(g["pp"], history) or []
        if not candidates:
            node["verdict"] = "DEAD_NO_PROPOSAL"
            dead += 1
            break
        advanced = False
        for tactic in candidates:
            risk = gate_tactic(tactic)
            ev = {"tool": "lean.run_tactic", "goal_id": g["goal_id"], "tactic": tactic, "risk": risk,
                  "state_id": st["state_id"], "pre": sha(g["pp"]), "depth": depth}
            child = {"tactic": tactic, "children": [], "verdict": "OPEN"}
            node["children"].append(child)
            if risk not in GATE_OK:
                ev.update(ok=False, reason=risk, post=None)
                events.append(ev)
                child["verdict"] = f"DEAD_{risk}"
                dead += 1
                if risk == "BLOCKED":          # fail-closed: stop the whole search
                    events.append({"tool": "lean.finalize", "no_goals": False, "reason": "BLOCKED_TACTIC"})
                    return {"events": events, "tree": tree, "done": False, "script": script,
                            "stats": {"steps": steps, "dead": dead, "depth": depth}}
                continue
            nxt = transport.step(st, g["goal_id"], tactic)
            steps += 1
            ev.update(ok=nxt["ok"], post=sha([x["pp"] for x in nxt["goals"]]) if nxt["ok"] else None,
                      message=nxt.get("message"), goals_after=len(nxt["goals"]) if nxt["ok"] else None)
            events.append(ev)
            if nxt["ok"]:
                history.append((g["pp"], tactic))
                script.append(tactic)
                st, node, advanced = nxt, child, True
                depth += 1
                break
            child["verdict"] = "DEAD_KERNEL"
            dead += 1
            if steps >= max_steps:
                break
        if not advanced:
            node["verdict"] = node.get("verdict") if node.get("verdict") != "OPEN" else "DEAD_EXHAUSTED"
            break
    if not st["goals"]:
        done = True
        node["verdict"] = "CLOSED"
    events.append({"tool": "lean.finalize", "no_goals": done, "steps": steps, "dead_branches": dead})
    return {"events": events, "tree": tree, "done": done, "script": script,
            "stats": {"steps": steps, "dead": dead, "depth": depth}}


def validate_proof_trace(events: list[dict], transport: Transport) -> dict:
    calls, bad = [], []
    for e in events:
        t = e.get("tool")
        if t == "lean.run_tactic":
            r = e.get("reason")
            if r == "BLOCKED":
                bad.append("BLOCKED_TACTIC_PRESENT")
            elif r == "EVIDENCE_ONLY":
                bad.append("EVIDENCE_IN_COMPLIANCE_LANE")
        if t == "lean.finalize" and e.get("no_goals") is not True:
            bad.append("GOALS_REMAIN")
        if t == "lean.initialize" and e.get("ok") is False:
            bad.append("INIT_FAILED")
        calls.append({"tool": t, "event_sha256": sha(e)})
    bad = sorted(set(bad))
    if bad:
        verdict = "VOID"
    elif not transport.certified:
        verdict = "TRACE_SIGNED_KERNEL_UNVERIFIED"
    else:
        verdict = "KERNEL_VERIFIED_AND_SIGNED"
    return {"events": calls, "violations": bad, "trace_sha256": sha(calls), "verdict": verdict,
            "kernel_verified": verdict == "KERNEL_VERIFIED_AND_SIGNED"}


def render_proof_file(prelude: str, name: str, statement: str, script: list[str], guard: bool = True) -> str:
    body = "\n".join(f"  {t}" for t in script) if script else "  sorry"
    out = "import GEH.Compliance.GehGuard\n\n"
    if prelude.strip():
        out += prelude.rstrip() + "\n\n"
    out += f"theorem {name} : {statement} := by\n{body}\n"
    if guard:
        out += f"\n#geh_guard {name}\n"
    return out


# ----------------------------------------------------------------------------
# Randomized-syntax crypto-property suite (one semantics, K surfaces).
# Oracles are computed in Python; the kernel must agree by rfl/decide.
# ----------------------------------------------------------------------------
PRELUDE = (
    "def gehPae (pt payload : String) : String :=\n"
    '  "DSSEv1 " ++ toString pt.length ++ " " ++ pt ++ " " ++ toString payload.length ++ " " ++ payload\n\n'
    "def gehLink (prev m mod : Nat) : Nat := (prev * 31 + m) % mod\n"
)


def py_pae(pt: str, payload: str) -> str:
    return f"DSSEv1 {len(pt)} {pt} {len(payload)} {payload}"


def py_link(prev: int, m: int, mod: int) -> int:
    return (prev * 31 + m) % mod


def gen_crypto_suite(run_id: str, salt: str, k: int = 6) -> dict:
    rng = random.Random(f"{run_id}:{salt}:crypto")
    alpha = "abcdxyz"
    items = []
    for i in range(k):
        can = canary(run_id, "C", i)
        tag = can[-8:].replace("-", "")
        arrow = rng.choice(["→", "->"])
        sp = rng.choice(["", " "])
        fam = i % 5
        if fam == 0:  # PAE framing (closed)
            pt = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 3)))
            pl = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 4)))
            stmt = f'gehPae "{pt}" "{pl}" = "{py_pae(pt, pl)}"'
            script = ["decide"]
            fam_name = "pae_framing"
        elif fam == 1:  # framing unambiguity (closed)
            a, b = "".join(rng.choice(alpha) for _ in range(2)), rng.choice(alpha)
            stmt = f'gehPae "{a}" "{b}" ≠ gehPae "{a[0]}" "{a[1]}{b}"'
            script = ["decide"]
            fam_name = "pae_unambiguous"
        elif fam == 2:  # xor self-inverse (structural, binder names canaryed)
            x, y = f"b{tag[:3]}", f"c{tag[3:6]}"
            stmt = f"∀ ({x} {y} : Bool),{sp} Bool.xor (Bool.xor {x} {y}) {y} = {x}"
            script = [f"intro {x} {y}", f"cases {x} <;> cases {y} <;> rfl"]
            fam_name = "xor_otp_involution"
        elif fam == 3:  # hash-chain step (closed, baked constants)
            mod = rng.choice([7919, 997, 65521])
            p, m = rng.randrange(1, 100000), rng.randrange(1, 100000)
            stmt = f"gehLink {p} {m} {mod} = {py_link(p, m, mod)}"
            script = ["rfl"]
            fam_name = "hash_chain_step"
        else:  # nonce uniqueness (closed)
            ns = sorted(rng.sample(range(1, 500), 4))
            stmt = f"{ns}.eraseDups = {ns}"
            script = ["decide"]
            fam_name = "nonce_nodup"
        # implication-wrapped variant: p → stmt (surface variance, same content)
        wrap = rng.random() < 0.4 and fam in (0, 3)
        if wrap:
            h = f"h{tag[:4]}"
            stmt_full = f"∀ ({h} : True),{sp} {stmt}"
            script = [f"intro {h}"] + script
        else:
            stmt_full = stmt
        stmt_full = stmt_full.replace("→", arrow) if arrow == "->" else stmt_full
        items.append({"id": f"crypto-r{i:02d}", "family": fam_name, "name": f"geh_{tag}", "statement": stmt_full,
                      "reference_script": script, "canary": can})
    text = f"/- GEH crypto suite run={run_id} salt={salt}. AUTO-GENERATED; do not hand-edit. -/\n"
    text += "import GEH.Compliance.GehGuard\n\n" + PRELUDE + "\n"
    for it in items:
        text += f"theorem {it['name']} : {it['statement']} := by\n" + "\n".join(f"  {t}" for t in it["reference_script"]) + "\n"
        text += f"#geh_guard {it['name']}\n\n"
    return {"generator": "geh-crypto-syntax/v2", "run_id": run_id, "salt": salt, "item_count": len(items),
            "items": items, "text": text, "digest": sha([it["statement"] for it in items]),
            "canaries": [it["canary"] for it in items]}


def surface_variants(statement: str) -> list[str]:
    """Purely syntactic realizations that MUST elaborate to the same goal."""
    v = [statement, statement.replace("→", "->"), re.sub(r"\s+", " ", statement),
         statement.replace(" = ", "  =  ")]
    return list(dict.fromkeys(v))


def contamination_score(outputs: list[str], canaries: list[str]) -> dict:
    """Foreign-canary leakage across ALL outputs (v4 compared against an empty list)."""
    hits = [{"output": i, "canary_from": j} for i, out in enumerate(outputs)
            for j, c in enumerate(canaries) if j != i and c in out]
    return {"leak_events": hits, "contaminated": bool(hits), "outputs_scored": len(outputs)}


# ----------------------------------------------------------------------------
# Chain, Article-12 log, provenance pins
# ----------------------------------------------------------------------------
GENESIS = hashlib.sha256(b"GEH-GENESIS").hexdigest()


def chain_append(paths: Paths, receipt_name: str, receipt_sha: str) -> str:
    prev = GENESIS
    if paths.chain.exists() and paths.chain.read_text(encoding="utf-8").strip():
        lines = paths.chain.read_text(encoding="utf-8").strip().splitlines()
        if not json.loads(lines[0]).get("genesis"):
            raise SystemExit("FAIL-CLOSED: chain missing genesis")
        prev = json.loads(lines[-1])["sha256"]
    else:
        with open(paths.chain, "w", encoding="utf-8") as f:
            f.write(canon({"genesis": True, "sha256": prev}) + "\n")
    with open(paths.chain, "a", encoding="utf-8") as f:
        f.write(canon({"receipt": receipt_name, "sha256": receipt_sha, "prev": prev, "ts": utcnow()}) + "\n")
    return prev


def verify_chain(paths: Paths) -> dict:
    lines = [json.loads(l) for l in paths.chain.read_text(encoding="utf-8").strip().splitlines()]
    if not lines or not lines[0].get("genesis"):
        raise SystemExit("BROKEN CHAIN: missing genesis")
    prev, n = lines[0]["sha256"], 0
    pub = load_pub(paths.keydir) if HAS_ED and (paths.keydir / "geh_ed25519.pub").exists() else None
    for link in lines[1:]:
        if link["prev"] != prev:
            raise SystemExit(f"BROKEN LINK at {link['receipt']}")
        p = paths.evidence / link["receipt"]
        env = json.loads(p.read_text(encoding="utf-8"))
        if sha_bytes(canon(env).encode("utf-8")) != link["sha256"]:
            raise SystemExit(f"RECEIPT BYTES DRIFTED: {link['receipt']}")
        if pub is not None and not DSSESigner.verify(env, pub):
            raise SystemExit(f"BAD SIGNATURE: {link['receipt']}")
        prev, n = link["sha256"], n + 1
    return {"receipts": n, "head": prev}


def emit_a12(paths: Paths, evt: dict) -> None:
    with open(paths.a12log, "a", encoding="utf-8") as f:
        f.write(canon({"regulation": "EU 2024/1689", "article": "12",
                       "note": "engineering mapping of Art.12 record-keeping; not legal advice", **evt}) + "\n")


def toolchain_pin(lean_project: Path) -> dict:
    pin = {"expected_toolchain": LEAN_TOOLCHAIN, "repl_rev": REPL_REV}
    tc = lean_project / "lean-toolchain"
    pin["lean_toolchain"] = tc.read_text(encoding="utf-8").strip() if tc.exists() else "NOT_FOUND"
    pin["toolchain_match"] = pin["lean_toolchain"] == LEAN_TOOLCHAIN
    man = lean_project / "lake-manifest.json"
    pin["lake_manifest_sha256"] = sha_file(man) if man.exists() else "NOT_FOUND"
    guard = lean_project / "GEH" / "Compliance" / "GehGuard.lean"
    pin["guard_sha256"] = sha_file(guard) if guard.exists() else "NOT_FOUND"
    try:
        v = subprocess.run(["lean", "--version"], capture_output=True, text=True, timeout=20, cwd=str(lean_project))
        pin["lean_version_reported"] = (v.stdout or v.stderr).strip()[:120]
    except Exception:
        pin["lean_version_reported"] = "UNAVAILABLE"
    return pin


def environment_pin() -> dict:
    me = Path(__file__).resolve()
    return {"harness_file": me.name, "harness_sha256": sha_file(me), "python": platform.python_version(),
            "platform": platform.platform(), "cryptography": (__import__("cryptography").__version__ if HAS_ED else "ABSENT"),
            "hostname_sha256": sha_bytes(platform.node().encode())[:16]}


# ----------------------------------------------------------------------------
# Lean project emission (the whole lake project, not a stub)
# ----------------------------------------------------------------------------
LEAN_FILES: dict[str, str] = {}  # filled from the sibling geh-lean directory at emit time


def emit_lean_project(dst: Path, src: Optional[Path] = None) -> list[str]:
    src = src or (Path(__file__).resolve().parent / "geh-lean")
    if not (src / "GEH" / "Compliance" / "GehGuard.lean").exists():
        raise SystemExit(f"FAIL-CLOSED: Lean sources not found next to the harness at {src}")
    written = []
    for p in src.rglob("*"):
        if p.is_dir() or ".lake" in p.parts or p.name == "lake-manifest.json":
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(p.read_bytes())
        written.append(str(rel))
    guard = (dst / "GEH" / "Compliance" / "GehGuard.lean").read_text(encoding="utf-8")
    assert "#geh_guard" in guard and len(guard) > 1000, "FAIL-CLOSED: guard source must be real"
    return sorted(written)


def repl_binary(lean_project: Path) -> Optional[Path]:
    """Locate the REPL executable built by `lake build repl` (package dir is
    `REPL` on case-sensitive filesystems, `repl` was the v7 assumption)."""
    pk = lean_project / ".lake" / "packages"
    if not pk.exists():
        return None
    for d in pk.iterdir():
        if d.name.lower() == "repl":
            for cand in (d / ".lake" / "build" / "bin" / "repl", d / ".lake" / "build" / "bin" / "repl.exe"):
                if cand.exists():
                    return cand
    return None


def repl_available(lean_project: Path) -> bool:
    return repl_binary(lean_project) is not None


# ----------------------------------------------------------------------------
# Self-check: structural (always) + kernel-grounded (when the REPL is built)
# ----------------------------------------------------------------------------
def selfcheck(paths: Paths, with_kernel: bool = True) -> dict:
    checks: dict[str, Any] = {}
    sim = SimulatedTransport()
    r = prove(sim, "A ∧ B", ScriptedProver(["constructor", "rfl", "rfl", "rfl"]))
    tr = validate_proof_trace(r["events"], sim)
    checks["sim_loop_closes"] = r["done"] is True
    checks["sim_never_certifies"] = tr["verdict"] == "TRACE_SIGNED_KERNEL_UNVERIFIED"
    checks["branching_exercised"] = any(e.get("goals_after", 0) and e["goals_after"] >= 2 for e in r["events"] if e.get("tool") == "lean.run_tactic")
    rb = prove(sim, "P", ScriptedProver(["sorry"]))
    checks["sorry_blocked_and_void"] = validate_proof_trace(rb["events"], sim)["verdict"] == "VOID" and rb["done"] is False
    checks["combinator_sorry_blocked"] = gate_tactic("cases a <;> sorry") == "BLOCKED"
    checks["native_decide_blocked"] = gate_tactic("native_decide") == "BLOCKED"
    checks["plausible_isolated"] = gate_tactic("plausible") == "EVIDENCE_ONLY"
    re_ = prove(sim, "P", ScriptedProver(["plausible"]))
    checks["evidence_void_in_compliance"] = "EVIDENCE_IN_COMPLIANCE_LANE" in validate_proof_trace(re_["events"], sim)["violations"]
    checks["exact_h_not_unregistered"] = gate_tactic("exact hpq hp") == "OK"
    checks["classifier_quarantines_sorry"] = classify_lean_artifact("theorem t : p := by sorry") == "QUARANTINED"
    checks["classifier_quarantines_axiom"] = classify_lean_artifact("axiom foo : False\n") == "QUARANTINED"
    checks["classifier_ignores_comment_sorry"] = classify_lean_artifact("-- no sorry here\ntheorem t : True := trivial") == "COMPLIANCE_CANDIDATE"
    checks["classifier_evidence_plausible"] = classify_lean_artifact("-- GEH-EVIDENCE-ONLY\nimport Plausible\nexample : True := by plausible") == "EVIDENCE_ONLY"
    s1, s2 = gen_crypto_suite("runabc123", "s", 5), gen_crypto_suite("runabc123", "s", 5)
    checks["crypto_suite_deterministic"] = s1["digest"] == s2["digest"]
    checks["crypto_realizations_differ"] = gen_crypto_suite("runabc123", "s2", 5)["digest"] != s1["digest"]
    checks["crypto_oracle_consistent"] = py_link(123, 456, 7919) == 4269 and py_pae("ab", "c") == "DSSEv1 2 ab 1 c"
    checks["contamination_scoring_detects_leak"] = contamination_score(
        [f"x {canary('rid00001', 'C', 1)}", "clean"], [canary("rid00001", "C", 0), canary("rid00001", "C", 1)])["contaminated"]
    checks["contamination_scoring_clean_passes"] = not contamination_score(["a", "b"], ["c1", "c2"])["contaminated"]
    # a fake dojo exercises the LeanDojo wiring (NOT a kernel)
    class _PF: ...
    class _TS:
        def __init__(self, pp, id_): self.pp, self.id, self.message = pp, id_, None
    class _Dojo:
        def run_tac(self, s, t): return _PF() if t.startswith("exact") else _TS("h : p\n⊢ q", 2)
    dj = LeanDojoTransport(_Dojo(), _TS("⊢ p → q", 1), _PF)
    rd = prove(dj, "p → q", ScriptedProver(["intro h", "exact h"]))
    checks["leandojo_wiring_mocked"] = rd["done"] is True and dj.verified_here is False
    if HAS_ED:
        s = DSSESigner(paths.keydir)
        env = s.sign({"probe": 1})
        pub = load_pub(paths.keydir)
        checks["dsse_roundtrip"] = DSSESigner.verify(env, pub)
        tampered = dict(env, payload=env["payload"] + " ")
        checks["dsse_rejects_tamper"] = not DSSESigner.verify(tampered, pub)
    else:
        checks["dsse_roundtrip"] = False
        checks["dsse_rejects_tamper"] = False
    checks["kernel_available"] = with_kernel and repl_available(paths.lean_project)
    if checks["kernel_available"]:
        checks.update(kernel_selfcheck(paths))
    checks["self_check_pass"] = all(v is True for k, v in checks.items() if k not in ("kernel_available",))
    return checks


def kernel_selfcheck(paths: Paths) -> dict:
    """Negative and positive cases against the real kernel. Each is a claim the
    harness makes about itself; each is checked, not asserted."""
    out: dict[str, Any] = {}
    t = LeanReplTransport(paths.lean_project)
    try:
        # green: closes and audits clean
        r = prove(t, "∀ (p q : Prop), p → (p → q) → q", ScriptedProver(["intro p q hp hpq", "exact hpq hp"]), name="geh_k_green")
        out["kernel_green_closes"] = r["done"] is True
        f = paths.evidence / "selfcheck_green.lean"
        f.write_text(render_proof_file("", "geh_k_green", "∀ (p q : Prop), p → (p → q) → q", r["script"]), encoding="utf-8")
        a = t.audit_source(f.read_text(encoding="utf-8"), ["geh_k_green"])
        out["kernel_green_audit_clean"] = a["error_count"] == 0 and a["sorry_count"] == 0 and a["guard_all_allowed"]
        out["kernel_green_axioms_in_trust_base"] = all(set(g["axioms"] or []) <= set(ALLOWED_AXIOMS) for g in a["guard"])
        # red 1: wrong tactic is rejected by the kernel and recorded as a dead branch
        r2 = prove(t, "∀ (p q : Prop), p → q", ScriptedProver(["intro p q hp", "exact hp"]), name="geh_k_red1")
        out["kernel_rejects_wrong_proof"] = r2["done"] is False and any(e.get("ok") is False and e.get("tool") == "lean.run_tactic" for e in r2["events"])
        # red 2: sorry file — guard must FAIL the build (sorryAx)
        f2 = paths.evidence / "selfcheck_red_sorry.lean"
        f2.write_text("import GEH.Compliance.GehGuard\ntheorem geh_k_red2 (p : Prop) : p := by sorry\n#geh_guard geh_k_red2\n", encoding="utf-8")
        a2 = t.audit_source(f2.read_text(encoding="utf-8"), [])
        out["kernel_guard_fails_on_sorry"] = a2["error_count"] >= 1 and any("GEH GUARD FAILED" in e for e in a2["errors"])
        # red 3: native_decide — guard must FAIL the build (Lean.ofReduceBool)
        f3 = paths.evidence / "selfcheck_red_native.lean"
        f3.write_text("import GEH.Compliance.GehGuard\ntheorem geh_k_red3 : 2 ^ 10 = 1024 := by native_decide\n#geh_guard geh_k_red3\n", encoding="utf-8")
        a3 = t.audit_source(f3.read_text(encoding="utf-8"), [])
        out["kernel_guard_fails_on_native_decide"] = a3["error_count"] >= 1 and any("ofReduceBool" in e for e in a3["errors"])
        # red 4: foreign axiom — guard must FAIL the build
        f4 = paths.evidence / "selfcheck_red_axiom.lean"
        f4.write_text("import GEH.Compliance.GehGuard\naxiom gehForeign : ∀ (p : Prop), p\ntheorem geh_k_red4 : 1 = 2 := gehForeign _\n#geh_guard geh_k_red4\n", encoding="utf-8")
        a4 = t.audit_source(f4.read_text(encoding="utf-8"), [])
        out["kernel_guard_fails_on_foreign_axiom"] = a4["error_count"] >= 1 and any("gehForeign" in e for e in a4["errors"])
        # surface variance, elaborated invariance
        hashes = set()
        for v in surface_variants("∀ (p q : Prop), p → (p → q) → q"):
            s = t.start(v, name="geh_k_var")
            hashes.add(s.get("elaborated_goal_sha256"))
        out["surface_variants_elaborate_identically"] = len(hashes) == 1 and None not in hashes
        # backtracking: first candidate dies, second closes (immutable proof states)
        class _Two:
            def __init__(self): self.n = 0
            def propose(self, pp, h):
                self.n += 1
                return ["exact Nat.zero_lt_one", "rfl"] if self.n == 1 else []
        r5 = prove(t, "(2 : Nat) + 2 = 4", _Two(), name="geh_k_bt")
        out["kernel_backtracking_from_immutable_state"] = r5["done"] is True and r5["stats"]["dead"] == 1
        # timeout discipline: a runaway tactic becomes a dead branch and the REPL is respawned
        old, before = t.timeout, t.respawns
        t.timeout = 0.0                         # deadline already elapsed: deterministic timeout regardless of machine speed
        r6 = prove(t, "True", ScriptedProver(["trivial"]), name="geh_k_to")
        t.timeout = old
        out["kernel_timeout_is_dead_branch_not_hang"] = (r6["done"] is False) and t.respawns > before and t.proc is not None and t.proc.poll() is None
        r7 = prove(t, "True", ScriptedProver(["trivial"]), name="geh_k_recover")
        out["kernel_recovers_after_timeout"] = r7["done"] is True
    finally:
        t.close()
    return out


# ----------------------------------------------------------------------------
# Receipt assembly
# ----------------------------------------------------------------------------
def issue_receipt(paths: Paths, signer: DSSESigner, run_id: str, subject: dict, proof: dict, sc: dict,
                  extra: Optional[dict] = None) -> tuple[Path, str, dict]:
    payload = {
        "_type": RECEIPT_TYPE,
        "subject": subject,
        "proof": proof,
        "run": {"run_id": run_id, "generated_utc": utcnow(), "human_verifier": None},
        "harness": {"version": HV, "toolspec": TOOLSPEC_ID, "self_check": sc, "environment": environment_pin(),
                    "toolchain": toolchain_pin(paths.lean_project)},
        "decision": {"gate": "PASS" if proof.get("kernel_verified") else "VOID", "risk_flag": not proof.get("kernel_verified"),
                     "pass_requires": TOOLSPEC["pass_requires"]},
        "compliance": {"eu_ai_act": {"article": "12", "automatic_logging": True, "retention_months": 6,
                                     "disclaimer": "engineering mapping, not legal advice"}},
        "chain": {"prev_receipt_sha256": None},
    }
    if extra:
        payload.update(extra)
    rf = paths.evidence / f"geh_proof_receipt_{run_id}.json"
    if rf.exists():
        # Receipts are append-only evidence. Overwriting one would orphan its chain entry (the chain
        # records the old bytes' hash) and every later --verify-chain would report RECEIPT BYTES DRIFTED.
        # Observed on owner metal 2026-09-30 when two runbook passes shared a minute-granular run id.
        emit_a12(paths, {"event": "geh.receipt_collision_refused", "ts": utcnow(), "run_id": run_id, "path": rf.name})
        raise SystemExit(f"FAIL-CLOSED: receipt {rf.name} already exists; refusing to overwrite chained evidence. "
                         f"Use a fresh --run-id.") from None
    # two-phase: link first (needs the chain head), then sign the linked payload
    head = GENESIS
    if paths.chain.exists() and paths.chain.read_text(encoding="utf-8").strip():
        head = json.loads(paths.chain.read_text(encoding="utf-8").strip().splitlines()[-1])["sha256"]
    payload["chain"]["prev_receipt_sha256"] = head
    env = signer.sign(payload)
    rf.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    rsha = sha_bytes(canon(env).encode("utf-8"))
    chain_append(paths, rf.name, rsha)
    emit_a12(paths, {"event": "geh.proof", "ts": payload["run"]["generated_utc"], "run_id": run_id,
                     "verdict": proof.get("verdict"), "gate": payload["decision"]["gate"], "receipt_sha256": rsha})
    return rf, rsha, payload


def make_transport(kind: str, paths: Paths) -> Transport:
    if kind == "sim":
        return SimulatedTransport()
    if kind == "repl":
        if not repl_available(paths.lean_project):
            raise SystemExit(f"FAIL-CLOSED: Lean REPL not built under {paths.lean_project}. Run: lake build && lake build repl")
        return LeanReplTransport(paths.lean_project)
    if kind == "pantograph":
        try:
            return PantographTransport(paths.lean_project)
        except ImportError:
            raise SystemExit("FAIL-CLOSED: PyPantograph not installed (git clone --recurse-submodules stanford-centaur/PyPantograph; uv sync). No PASS possible without a kernel.")
    raise SystemExit(f"unknown transport {kind}")


def run_prove(paths: Paths, transport: Transport, statement: str, prover, *, name: str, prelude: str = "") -> dict:
    """Live channel + audit channel + guard = one proof object."""
    r = prove(transport, statement, prover, prelude=prelude, name=name)
    tr = validate_proof_trace(r["events"], transport)
    proof = {"verdict": tr["verdict"], "transport": transport.name, "transport_verified_here": transport.verified_here,
             "live": {"done": r["done"], "stats": r["stats"], "trace_sha256": tr["trace_sha256"], "violations": tr["violations"],
                      "tree_sha256": sha(r["tree"]), "script": r["script"], "events": r["events"]},
             "audit": None, "kernel_verified": False}
    if r["done"] and isinstance(transport, LeanReplTransport):
        src = render_proof_file(prelude, name, statement, r["script"])
        grade = classify_lean_artifact(src)
        f = paths.evidence / f"{name}.lean"
        f.write_text(src, encoding="utf-8")
        a = transport.audit_source(src, [name])
        a["artifact_grade"], a["file"], a["file_sha256"] = grade, f.name, sha_file(f)
        proof["audit"] = a
        ok = (tr["kernel_verified"] and a["error_count"] == 0 and a["sorry_count"] == 0 and a["guard_all_allowed"]
              and grade == "COMPLIANCE_CANDIDATE")
        proof["kernel_verified"] = ok
        if not ok:
            proof["verdict"] = "VOID"
            proof["void_reasons"] = [x for x, c in [("LIVE_TRACE", not tr["kernel_verified"]), ("AUDIT_ERRORS", a["error_count"] > 0),
                                                    ("AUDIT_SORRIES", a["sorry_count"] > 0), ("GUARD_REJECTED", not a["guard_all_allowed"]),
                                                    (f"ARTIFACT_{grade}", grade != "COMPLIANCE_CANDIDATE")] if c]
    elif r["done"] and transport.certified:
        # certified transport without an audit channel wired in this build: never PASS
        proof["verdict"] = "TRACE_SIGNED_AUDIT_CHANNEL_MISSING"
    return proof


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SZL GEH v8 — governed Lean 4 verification harness")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--lean-project", type=Path, default=None, help="lake project dir (default ROOT/geh-lean)")
    ap.add_argument("--run-id", default=secrets.token_hex(8))
    ap.add_argument("--salt", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--no-kernel", action="store_true", help="skip kernel self-checks even if the REPL is built")
    ap.add_argument("--emit-lean", action="store_true")
    ap.add_argument("--emit-toolspec", action="store_true")
    ap.add_argument("--gen-crypto", type=int, default=0)
    ap.add_argument("--bench-crypto", type=int, default=0)
    ap.add_argument("--transport", choices=["sim", "repl", "pantograph"], default="sim")
    ap.add_argument("--prove")
    ap.add_argument("--script", help="comma-separated tactic script for --prove (else heuristic prover)")
    ap.add_argument("--audit-file", type=Path)
    ap.add_argument("--guard", action="append", default=[], help="theorem names to #geh_guard in --audit-file")
    ap.add_argument("--verify", type=Path)
    ap.add_argument("--verify-chain", action="store_true")
    ap.add_argument("--dirhash", type=Path, help="print the chaski runner's adapter dir-hash: sha256(name+NUL+bytes) over sorted *.safetensors")
    a = ap.parse_args(argv)

    if a.dirhash:
        h = hashlib.sha256()
        files = sorted(Path(a.dirhash).glob("*.safetensors"))
        if not files:
            raise SystemExit(f"no safetensors in {a.dirhash}")
        for f in files:
            h.update(f.name.encode("utf-8")); h.update(b"\0"); h.update(f.read_bytes())
        print(h.hexdigest())
        return 0

    root = a.root or default_root()
    paths = Paths(root=root, lean_project=a.lean_project or (root / "geh-lean"))

    if a.verify_chain:
        print(json.dumps({"CHAIN_VERIFIED": verify_chain(paths)}))
        return 0
    if a.verify:
        env = json.loads(a.verify.read_text(encoding="utf-8"))
        ok = DSSESigner.verify(env, load_pub(paths.keydir))
        print("VALID" if ok else "INVALID")
        return 0 if ok else 2
    if a.emit_toolspec:
        p = paths.evidence / "geh_tooluse_spec.v2.json"
        p.write_text(json.dumps(TOOLSPEC, indent=2), encoding="utf-8")
        print(f"EMITTED={p}")
        return 0
    if a.emit_lean:
        files = emit_lean_project(paths.lean_project)
        print(json.dumps({"EMITTED": str(paths.lean_project), "files": files}, indent=2))
        return 0
    if a.gen_crypto:
        s = gen_crypto_suite(a.run_id, a.salt, a.gen_crypto)
        p = paths.evidence / f"geh_crypto_{a.run_id}.lean"
        p.write_text(s["text"], encoding="utf-8")
        print(json.dumps({"path": str(p), "digest": s["digest"], "canaries": s["canaries"], "items": [i["statement"] for i in s["items"]]}, indent=2, ensure_ascii=False))
        return 0
    if a.selfcheck:
        sc = selfcheck(paths, with_kernel=not a.no_kernel)
        print(json.dumps(sc, indent=2))
        return 0 if sc["self_check_pass"] else 1

    # every receipt-issuing path re-runs the self-check and refuses on failure
    sc = selfcheck(paths, with_kernel=(a.transport == "repl") and not a.no_kernel)
    if not sc["self_check_pass"]:
        emit_a12(paths, {"event": "geh.self_check_failed", "ts": utcnow(), "failed": [k for k, v in sc.items() if v is not True]})
        print(json.dumps(sc, indent=2))
        raise SystemExit("FAIL-CLOSED: self-check failed; no receipt issued.")

    signer = DSSESigner(paths.keydir)
    if a.audit_file:
        if a.transport != "repl":
            raise SystemExit("FAIL-CLOSED: --audit-file requires --transport repl")
        t = make_transport("repl", paths)
        try:
            src = a.audit_file.read_text(encoding="utf-8")
            grade = classify_lean_artifact(src)
            guards = a.guard or re.findall(r"^#geh_guard\s+(\S+)", src, flags=re.MULTILINE)
            au = t.audit_file(a.audit_file.resolve(), guards)
        finally:
            t.close()
        ok = grade == "COMPLIANCE_CANDIDATE" and au["error_count"] == 0 and au["sorry_count"] == 0 and au["guard_all_allowed"]
        proof = {"verdict": "KERNEL_VERIFIED_AND_SIGNED" if ok else "VOID", "transport": "LEAN_REPL_AUDIT", "transport_verified_here": True,
                 "kernel_verified": ok, "audit": dict(au, artifact_grade=grade, file=str(a.audit_file), file_sha256=sha_file(a.audit_file))}
        subject = {"kind": "lean_file", "file": a.audit_file.name, "file_sha256": sha_file(a.audit_file), "theorems": guards}
    elif a.bench_crypto:
        t = make_transport(a.transport, paths)
        suite = gen_crypto_suite(a.run_id, a.salt, a.bench_crypto)
        results, outputs = [], []
        try:
            for it in suite["items"]:
                if isinstance(t, LeanReplTransport) and t.rpc_count >= 40:
                    t._respawn()               # bound REPL memory: environments accumulate
                prover = ScriptedProver(it["reference_script"]) if a.transport == "sim" else HeuristicProver()
                # a real LLM goes here: CallableProposer(lambda pp, hist: your_model(pp, hist))
                pr = run_prove(paths, t, it["statement"], prover, name=it["name"], prelude=PRELUDE)
                results.append({"id": it["id"], "family": it["family"], "statement_sha256": sha(it["statement"]),
                                "verdict": pr["verdict"], "kernel_verified": pr["kernel_verified"], "stats": pr["live"]["stats"],
                                "script": pr["live"]["script"], "audit_tactics": (pr["audit"] or {}).get("tactic_count")})
                outputs.append(json.dumps(pr["live"]["script"]))
        finally:
            t.close()
        k_pass = sum(1 for r in results if r["kernel_verified"])
        cont = contamination_score(outputs, suite["canaries"])
        proof = {"verdict": "KERNEL_VERIFIED_AND_SIGNED" if (k_pass == len(results) and t.certified) else ("VOID" if t.certified else "TRACE_SIGNED_KERNEL_UNVERIFIED"),
                 "transport": t.name, "transport_verified_here": t.verified_here, "kernel_verified": t.certified and k_pass == len(results),
                 "bench": {"k": len(results), "pass": k_pass, "pass_at_k": round(k_pass / max(1, len(results)), 3),
                           "steps_total": sum(r["stats"]["steps"] for r in results), "dead_total": sum(r["stats"]["dead"] for r in results),
                           "contamination": cont, "suite_digest": suite["digest"], "generator": suite["generator"]},
                 "results": results}
        subject = {"kind": "crypto_suite", "run_id": a.run_id, "salt": a.salt, "suite_digest": suite["digest"]}
        (paths.evidence / f"geh_crypto_{a.run_id}.lean").write_text(suite["text"], encoding="utf-8")
    elif a.prove:
        t = make_transport(a.transport, paths)
        try:
            prover = ScriptedProver([s.strip() for s in a.script.split(",")]) if a.script else HeuristicProver()
            proof = run_prove(paths, t, a.prove, prover, name=f"geh_{a.run_id[:8]}", prelude=PRELUDE if "geh" in a.prove else "")
        finally:
            t.close()
        subject = {"kind": "statement", "statement": a.prove, "statement_sha256": sha(a.prove)}
    else:
        ap.print_help()
        raise SystemExit("FAIL-CLOSED: choose --prove / --bench-crypto / --audit-file / --selfcheck")

    rf, rsha, payload = issue_receipt(paths, signer, a.run_id, subject, proof, sc)
    print(f"VERDICT={proof['verdict']} TRANSPORT={proof['transport']} GATE={payload['decision']['gate']}")
    print(f"RECEIPT={rf}\nRECEIPT_SHA256={rsha}")
    print("No training, adapter modification, upload, publication, deployment, or Git write occurred.")
    return 0 if payload["decision"]["gate"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
