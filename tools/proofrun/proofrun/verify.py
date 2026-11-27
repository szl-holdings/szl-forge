"""The claim checker. Fails closed: a claim passes only when every required check PASSes.

Integrity (applies to every claim): every envelope signature verifies, one anchored signer set,
one run_id, a pinned environment manifest, and (proof-carrying) an unbroken heartbeat chain.

steps claim      claimed == terminal checkpoint global_step == max_steps, re-derived from the real
                 trainer_state.json bytes, whose sha256 matches the signed receipt; the last heartbeat
                 is at that step and the checkpoint is bound to the heartbeat head at that step.
final claim      final adapter_model.safetensors is byte-equal (sha256) to the terminal checkpoint's,
                 not to any earlier checkpoint, and the adapter is not a LoRA no-op (all lora_B zero).
quality claim    each metric is backed by a signed eval receipt that names the terminal checkpoint tree
                 hash, the final adapter hash and the run's pinned image digest, and the value matches.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .canonical import ProofError, is_hex64, load_json, loads_strict, sha256_file, sha256_obj, tree_sha256
from .records import DIGEST_RE
from .safetensors_lite import lora_report
from .endorse import (ENDORSEMENT_FILE, ENDORSEMENT_SCHEME, GITHUB_ISSUER, STATEMENT_FILE, bundle_digest,
                      bundle_statement, check_scope, check_sigstore, statement_bytes)
from .signing import FULL_FP_RE, fp_matches, full_fingerprint, open_envelope, valid_fingerprint

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"
ADAPTER = "adapter_model.safetensors"


@dataclass
class Section:
    name: str
    checks: list = field(default_factory=list)

    def add(self, status: str, check: str, detail: str = "") -> None:
        self.checks.append({"status": status, "check": check, "detail": detail})

    def ok(self, cond: bool, check: str, detail: str = "") -> bool:
        self.add(PASS if cond else FAIL, check, detail)
        return cond

    @property
    def status(self) -> str:
        if any(c["status"] == FAIL for c in self.checks):
            return FAIL
        if not any(c["status"] == PASS for c in self.checks):
            return FAIL  # nothing proven is not a pass
        return PASS


class Verifier:
    def __init__(self, run_dir: Path | str, trusted: set[str] | None = None,
                 accept_retroactive: bool = False, trusted_identities: set[str] | None = None,
                 issuer: str = GITHUB_ISSUER, gh_runner=None) -> None:
        self.dir = Path(run_dir)
        self.trusted = set(trusted) if trusted else None
        self.identities = set(trusted_identities) if trusted_identities else None
        self.issuer, self.gh_runner = issuer, gh_runner
        self.accept_retroactive = accept_retroactive
        self.signers: set[str] = set()
        self.signer_keys: set[str] = set()
        self.run_id = None
        self.endorsement_scope = None

    # -- helpers -------------------------------------------------------------------
    def _open(self, env, kind, sec: Section, label: str):
        try:
            # trust anchoring is decided once, over the whole bundle, in _anchor()
            payload, fp = open_envelope(env, kind, None)
        except ProofError as e:
            sec.add(FAIL, f"signature {label}", str(e))
            return None
        self.signers.add(fp)
        self.signer_keys.add(env["sig"]["public_key"])
        return payload

    def _load(self, rel: str, kind: str, sec: Section, required: bool = True):
        p = self.dir / rel
        if not p.exists():
            if required:
                sec.add(FAIL, f"present {rel}", "missing (fail closed)")
            return None
        try:
            env = load_json(p)
        except ProofError as e:
            sec.add(FAIL, f"parse {rel}", str(e))
            return None
        return self._open(env, kind, sec, rel)

    def _artifact(self, adir: str | None, rel: str) -> Path | None:
        if not adir:
            return None
        base = (self.dir / adir).resolve()
        if self.dir.resolve() not in base.parents and base != self.dir.resolve():
            return None  # never follow artifact paths outside the run dir
        p = base / rel
        return p if p.is_file() else None

    def _rehash(self, rec: dict, sec: Section, label: str) -> dict:
        """Re-hash every listed file that is present; fail on mismatch or on unlisted extra files."""
        files = rec.get("files") or []
        sec.ok(rec.get("tree_sha256") == tree_sha256(files), f"{label} tree hash",
               "tree_sha256 recomputed from file list")
        present, adir = 0, rec.get("artifact_dir")
        for f in files:
            p = self._artifact(adir, f["path"])
            if p is None:
                continue
            present += 1
            got = sha256_file(p)
            if got != f["sha256"]:
                sec.add(FAIL, f"{label} file hash {f['path']}", f"on disk {got[:12]} != receipt {f['sha256'][:12]} (TAMPERED)")
        if adir and (self.dir / adir).is_dir():
            listed = {f["path"] for f in files}
            extra = [p.relative_to(self.dir / adir).as_posix() for p in (self.dir / adir).rglob("*")
                     if p.is_file() and p.relative_to(self.dir / adir).as_posix() not in listed]
            sec.ok(not extra, f"{label} no unlisted files", f"extra: {extra[:5]}" if extra else "")
        sec.add(INFO, f"{label} re-derived", f"{present}/{len(files)} files re-hashed from bytes on disk; "
                f"rest attested by signed receipt only")
        return {f["path"]: f["sha256"] for f in files}

    # -- main ----------------------------------------------------------------------
    def run(self) -> dict:
        integ = Section("integrity")
        steps, final, quality = Section("steps"), Section("final_adapter"), Section("quality")

        claim = self._load("claim.json", "claim", integ)
        man = self._load("run_manifest.json", "run_manifest", integ)
        if claim is None or man is None:
            return self._report(integ, steps, final, quality, claim, man)

        run_id = claim.get("run_id")
        self.run_id = run_id
        level = claim.get("evidence_level")
        retro = bool(man.get("retroactive"))
        integ.ok(man.get("run_id") == run_id, "manifest run_id matches claim", str(run_id))

        # environment manifest
        if retro or level == "retroactive":
            integ.ok(self.accept_retroactive and retro and level == "retroactive",
                     "retroactive evidence accepted",
                     "run predates proofrun: no image digest, lockfile, dataset hash or heartbeat chain from the "
                     "original run" + ("" if self.accept_retroactive else "; rerun with --accept-retroactive to evaluate"))
        else:
            img = (man.get("image") or {})
            integ.ok(isinstance(img.get("ref"), str) and bool(DIGEST_RE.match(img["ref"]))
                     and img.get("digest") == "sha256:" + img["ref"].rsplit("sha256:", 1)[-1],
                     "image pinned by digest", str(img.get("ref")))
            integ.ok(is_hex64(man.get("lockfile_sha256")), "lockfile hashed")
            integ.ok(is_hex64((man.get("dataset") or {}).get("sha256")), "dataset hashed")
            integ.ok(man.get("config_sha256") == sha256_obj(man.get("config")), "config hash matches config")

        # heartbeat chain
        beats = self._chain(integ, run_id, retro)

        # checkpoints
        ckpts = []
        for p in sorted((self.dir / "checkpoints").glob("*.json")):
            try:
                env = load_json(p)
            except ProofError as e:
                integ.add(FAIL, f"parse {p.name}", str(e)); continue
            pl = self._open(env, "checkpoint_receipt", integ, f"checkpoints/{p.name}")
            if pl is not None:
                ckpts.append(pl)
        bad = [c.get("checkpoint") for c in ckpts if c.get("run_id") != run_id]
        integ.ok(not bad, "checkpoint receipts signed and share run_id",
                 f"{len(ckpts)} receipts" + (f"; wrong run_id: {bad[:5]}" if bad else ""))
        ckpts.sort(key=lambda c: c.get("global_step", -1))
        term = ckpts[-1] if ckpts else None
        claimed = claim.get("steps")

        # ---- steps claim
        if not term:
            steps.add(FAIL, "terminal checkpoint receipt", "no checkpoint receipts (fail closed)")
        else:
            hashes = self._rehash(term, steps, term["checkpoint"])
            steps.ok(claimed == term["global_step"], "claimed steps == terminal global_step",
                     f"claimed {claimed}, checkpoint {term['checkpoint']} global_step {term['global_step']}")
            steps.ok(term["global_step"] == term.get("max_steps"), "run reached max_steps",
                     f"global_step {term['global_step']} / max_steps {term.get('max_steps')}")
            tsp = self._artifact(term.get("artifact_dir"), "trainer_state.json")
            if tsp is None:
                steps.add(FAIL, "trainer_state re-derived", "trainer_state.json bytes not present (fail closed)")
            else:
                got = sha256_file(tsp)
                steps.ok(got == term.get("trainer_state_sha256") == hashes.get("trainer_state.json"),
                         "trainer_state bytes match signed receipt", got[:16])
                try:
                    ts = loads_strict(tsp.read_text())
                    steps.ok(ts.get("global_step") == term["global_step"] == claimed,
                             "trainer_state.global_step (from bytes) == claim", str(ts.get("global_step")))
                    lh = [r.get("step") for r in ts.get("log_history", []) if isinstance(r.get("step"), int)]
                    steps.ok(bool(lh) and lh == sorted(lh) and lh[-1] == ts.get("global_step"),
                             "log_history monotonic and ends at global_step",
                             f"{len(lh)} entries, last step {lh[-1] if lh else None}")
                except (ProofError, json.JSONDecodeError) as e:
                    steps.add(FAIL, "trainer_state parse", str(e))
            if retro:
                steps.add(INFO, "heartbeat binding", "not available (retroactive run)")
            elif beats:
                heads = {b["_sha"]: b for b in beats}
                h = heads.get(term.get("heartbeat_head"))
                steps.ok(h is not None and h["step"] == term["global_step"],
                         "checkpoint bound to heartbeat at its step",
                         f"head step {h['step'] if h else None}")
                steps.ok(beats[-1]["step"] == claimed, "last heartbeat step == claim", str(beats[-1]["step"]))
            else:
                steps.add(FAIL, "heartbeat binding", "no heartbeat chain (fail closed)")

        # ---- final adapter claim
        fin = self._load("final.json", "final_adapter", final)
        fin_adapter = None
        if fin is not None and term is not None:
            integ.ok(fin.get("run_id") == run_id, "final run_id")
            fh = self._rehash(fin, final, "final")
            fin_adapter = fh.get(ADAPTER)
            th = {f["path"]: f["sha256"] for f in term["files"]}
            final.ok(fin_adapter is not None and fin_adapter == th.get(ADAPTER),
                     "final adapter byte-equal to terminal checkpoint",
                     f"final {str(fin_adapter)[:16]} vs {term['checkpoint']} {str(th.get(ADAPTER))[:16]}")
            stale = [c["checkpoint"] for c in ckpts[:-1]
                     if {f["path"]: f["sha256"] for f in c["files"]}.get(ADAPTER) == fin_adapter]
            final.ok(not stale, "final adapter is not an earlier checkpoint", f"equals {stale}" if stale else "")
            if "adapter_config.json" in fh and "adapter_config.json" in th:
                final.ok(fh["adapter_config.json"] == th["adapter_config.json"], "adapter_config equal to terminal")
            ap = self._artifact(fin.get("artifact_dir"), ADAPTER)
            if ap is None:
                final.add(FAIL, "LoRA no-op check", "adapter bytes not present (fail closed)")
            else:
                try:
                    rep = lora_report(ap)
                    final.ok(not rep["no_op"], "LoRA is not a no-op",
                             f"{rep['lora_B_tensors']} lora_B tensors, {rep['lora_B_nonzero']}/{rep['lora_B_elements']} nonzero")
                except ProofError as e:
                    final.add(FAIL, "LoRA no-op check", str(e))
        elif term is None:
            final.add(FAIL, "terminal checkpoint", "missing")

        # ---- quality claims
        qs = claim.get("quality") or []
        if not qs:
            quality.add(INFO, "quality", "no quality claim made")
        for q in qs:
            label = f"{q.get('metric')} {q.get('op')} {q.get('value')}"
            er = q.get("eval_receipt", "evals/MISSING.json")
            if not isinstance(er, str) or not (self.dir / er).resolve().is_relative_to(self.dir.resolve()):
                quality.add(FAIL, f"eval receipt path for {label}", "path escapes run dir (fail closed)")
                continue
            ev = self._load(er, "eval_receipt", quality)
            if ev is None:
                quality.add(FAIL, f"eval receipt for {label}", "no signed eval receipt (fail closed)")
                continue
            quality.ok(ev.get("run_id") == run_id, "eval run_id")
            quality.ok(term is not None and ev.get("checkpoint_tree_sha256") == term.get("tree_sha256"),
                       "eval names terminal checkpoint tree hash")
            quality.ok(fin_adapter is not None and ev.get("adapter_sha256") == fin_adapter,
                       "eval names final adapter hash")
            mdig = (man.get("image") or {}).get("digest")
            quality.ok(bool(mdig) and ev.get("image_digest") == mdig, "eval names the run's pinned image digest",
                       f"eval {ev.get('image_digest')} vs run {mdig}")
            quality.ok(is_hex64(ev.get("eval_set_sha256")) and is_hex64(ev.get("harness_sha256")),
                       "eval set and harness hashed")
            dsh = (man.get("dataset") or {}).get("sha256")
            quality.ok(is_hex64(dsh) and ev.get("eval_set_sha256") != dsh, "eval set is not the training dataset",
                       "needs the run's dataset hash (retroactive runs have none: held-out-ness unproven)")
            got = (ev.get("metrics") or {}).get(q.get("metric"))
            want = q.get("value")
            okv = (isinstance(got, (int, float)) and not isinstance(got, bool) and isinstance(want, (int, float))
                and not isinstance(want, bool) and math.isfinite(got) and (
                (q.get("op") == ">=" and got >= want) or (q.get("op") == "<=" and got <= want)
                or (q.get("op") == "==" and abs(got - want) <= 1e-9)))
            quality.ok(okv, f"metric {label}", f"receipt value {got}")

        return self._report(integ, steps, final, quality, claim, man)

    def _chain(self, integ: Section, run_id, retro: bool) -> list[dict]:
        p = self.dir / "heartbeats.jsonl"
        if not p.exists() or not p.read_text().strip():
            if not retro:
                integ.add(FAIL, "heartbeat chain present", "missing (fail closed)")
            else:
                integ.add(INFO, "heartbeat chain", "absent (retroactive run)")
            return []
        man_env = load_json(self.dir / "run_manifest.json")
        prev, out, ok = man_env.get("payload_sha256"), [], True
        last_step = -1
        for i, line in enumerate(p.read_text().splitlines()):
            try:
                env = loads_strict(line)
            except Exception as e:
                integ.add(FAIL, f"heartbeat {i} parse", str(e)); ok = False; break
            pl = self._open(env, "heartbeat", integ, f"heartbeat {i}")
            if pl is None:
                ok = False; break
            problems = []
            if pl.get("seq") != i: problems.append(f"seq {pl.get('seq')} != {i}")
            if pl.get("prev") != prev: problems.append("prev hash does not link (reordered, dropped or forged)")
            if pl.get("run_id") != run_id: problems.append("run_id")
            if pl.get("manifest_sha256") != man_env.get("payload_sha256"): problems.append("manifest binding")
            if not isinstance(pl.get("step"), int) or pl["step"] <= last_step:
                problems.append(f"step {pl.get('step')} not > {last_step}")
            if problems:
                integ.add(FAIL, f"heartbeat chain link {i}", "; ".join(problems)); ok = False; break
            last_step, prev = pl["step"], env["payload_sha256"]
            out.append(dict(pl, _sha=env["payload_sha256"]))
        if ok:
            integ.add(PASS, "heartbeat chain intact", f"{len(out)} links, steps {out[0]['step']}..{out[-1]['step']}, "
                      f"head {prev[:16]}")
        return out if ok else []

    def _anchor(self, integ: Section) -> None:
        """Every requested anchor must PASS (keys AND identities when both are given). Fail closed."""
        full = sorted(full_fingerprint(k) for k in self.signer_keys)
        if not full:
            integ.add(FAIL, "trust anchor", "no verified signer (fail closed)")
            return
        st = None

        def statement():
            nonlocal st
            if st is None:
                st = bundle_statement(self.dir)
            return st

        try:
            if (self.dir / STATEMENT_FILE).exists():
                integ.ok((self.dir / STATEMENT_FILE).read_bytes() == statement_bytes(statement()),
                         "shipped proofrun_bundle.json equals recomputed statement")
        except ProofError as e:
            integ.add(FAIL, "bundle statement", str(e)); return
        if self.trusted is not None:
            bad = sorted(k for k in self.trusted if not valid_fingerprint(k))
            if bad:
                integ.add(FAIL, "trusted key format",
                          f"{len(bad)} --trusted-key value(s) not ed25519:<16hex> or ed25519:sha256:<64hex>")
                return
            if all(fp_matches(k, self.trusted) for k in self.signer_keys):
                integ.add(PASS, "signers anchored to --trusted-key (direct)", str(full))
            else:
                self._anchor_endorsement(integ, full, statement)
        if self.identities is not None:
            try:
                s = statement()
            except ProofError as e:
                integ.add(FAIL, "bundle statement", str(e)); return
            if len(full) != 1 or s["run_signers"] != full or s["run_id"] != self.run_id:
                integ.add(FAIL, "keyless endorsement", "bundle must have exactly one run signer and match run_id")
                return
            detail = []
            for ident in sorted(self.identities):
                ok, d = check_sigstore(self.dir, s, ident, self.issuer, self.gh_runner)
                if ok:
                    integ.add(PASS, "endorsed by --trusted-identity (Sigstore keyless)", d)
                    break
                detail.append(d)
            else:
                integ.add(FAIL, "endorsed by --trusted-identity (Sigstore keyless)", "; ".join(detail))

    def _anchor_endorsement(self, integ: Section, full: list, statement) -> None:
        label = "endorsed by --trusted-key (org key)"
        p = self.dir / ENDORSEMENT_FILE
        if not p.exists():
            integ.add(FAIL, label, f"run signer {full} not trusted and {ENDORSEMENT_FILE} missing (fail closed)")
            return
        try:
            pl, efp = open_envelope(load_json(p), "endorsement", self.trusted)
            env_pk = load_json(p)["sig"]["public_key"]
            s = statement()
        except ProofError as e:
            integ.add(FAIL, label, str(e)); return
        problems = []
        if len(full) != 1: problems.append(f"expected exactly one run signer, got {full}")
        if pl.get("scheme") != ENDORSEMENT_SCHEME: problems.append("scheme")
        if pl.get("run_id") != self.run_id or s["run_id"] != self.run_id: problems.append("run_id")
        if pl.get("bundle_sha256") != bundle_digest(s):
            problems.append("bundle digest mismatch: receipts changed after endorsement (TAMPERED)")
        if pl.get("run_signers") != full or s["run_signers"] != full:
            problems.append("endorsed run signer does not match the bundle signer")
        if env_pk in self.signer_keys: problems.append("endorser is also a run signer")
        scope = pl.get("scope")
        if scope is not None:
            sp = check_scope(scope)
            if sp: problems.append(sp)
        if problems:
            integ.add(FAIL, label, "; ".join(problems)); return
        efull = full_fingerprint(env_pk)
        integ.add(PASS, label, f"endorser {efull}; run signer {full[0]}; bundle {pl['bundle_sha256'][:16]}")
        self.endorsement_scope = scope
        if scope is None:
            integ.add(INFO, "endorsement scope", "none stated: custody and integrity only; no claim endorsed")
        else:
            integ.add(INFO, "endorsement scope",
                      f"custody + {scope['endorsed_claims']}; NOT endorsed {scope['not_endorsed']} "
                      f"(at endorsement {scope['claim_status_at_endorsement']}); {scope['qualification']}")
        if not any(FULL_FP_RE.match(k) for k in self.trusted if fp_matches(env_pk, {k})):
            integ.add(INFO, "trusted key form", "64-bit short fingerprint used; prefer ed25519:sha256:<64hex>")

    def _report(self, integ, steps, final, quality, claim, man) -> dict:
        if self.trusted is None and self.identities is None:
            integ.ok(len(self.signers) == 1, "single signer (unanchored)",
                     f"signers {sorted(self.signers)}; compare fingerprint out of band or pass --trusted-key")
        elif claim is not None and man is not None:
            self._anchor(integ)
        base = integ.status
        claims = {}
        for s in (steps, final, quality):
            st = s.status if base == PASS else FAIL
            if s is quality and not (claim or {}).get("quality") and base == PASS:
                st = "NOT CLAIMED"
            claims[s.name] = {"status": st, "checks": s.checks}
        verdict = PASS if base == PASS and all(c["status"] in (PASS, "NOT CLAIMED") for c in claims.values()) else FAIL
        return {"verdict": verdict, "run_dir": str(self.dir), "run_id": (claim or {}).get("run_id"),
                "evidence_level": (claim or {}).get("evidence_level"),
                "signers": sorted(self.signers), "endorsement_scope": self.endorsement_scope, "integrity": {"status": base, "checks": integ.checks},
                "claims": claims}


def verify(run_dir, trusted=None, accept_retroactive=False, trusted_identities=None,
           issuer=GITHUB_ISSUER, gh_runner=None) -> dict:
    return Verifier(run_dir, trusted, accept_retroactive, trusted_identities, issuer, gh_runner).run()


def render(rep: dict) -> str:
    lines = [f"proofrun verify {rep['run_dir']}",
             f"  run_id={rep['run_id']} evidence_level={rep['evidence_level']} signers={rep['signers']}"]
    sections = [("integrity", rep["integrity"])] + list(rep["claims"].items())
    for name, sec in sections:
        lines.append(f"[{sec['status']}] {name}")
        for c in sec["checks"]:
            d = f"  ({c['detail']})" if c["detail"] else ""
            lines.append(f"    {c['status']:4} {c['check']}{d}")
    lines.append(f"VERDICT: {rep['verdict']}")
    return "\n".join(lines)
