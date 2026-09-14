"""Actual Second Brain -> Nemo -> recurrent inference -> Nemo research adapter.

Public evidence binds cache validity; it is NOT used as register training data or
claimed to improve the synthetic model. This library supplies no authentication
server: an enclosing trusted controller must supply the authorizer. No tool,
provider-write, private-content hydration or publication path exists here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import threading
from typing import Callable


from .runtime import Scope, Session, checked_digest, canonical, digest
from .training import LABEL_OFFSET, oracle, weight_bytes

PINS = {
    "szl-nemo": "158fd7e072da1086ff15405dda5df1718de1ff22",
    "szl-second-brain": "25c12301e2fdcb8876352f6566dc2f4d2c6710fc",
}


class BoundaryError(ValueError):
    """Evidence, identity, authorization or witness failed; do not generate."""


def nemo_digest(value: object) -> str:
    # The installed Nemo and Second Brain interfaces use UTF-8, not ASCII escaping.
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode())


def verified_install(name: str) -> dict[str, str]:
    distribution = metadata.distribution(name)
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != PINS[name]:
        raise BoundaryError(f"{name} must be installed from its admitted exact Git revision")
    return {"version": distribution.version, "revision": PINS[name]}


def evidence_binding(hit: dict) -> dict:
    if type(hit) is not dict or hit.get("ready") is not True or hit.get("content_access") != "HANDLES_ONLY":
        raise BoundaryError("no ready handles-only evidence")
    items, handles = hit.get("evidence"), hit.get("handles")
    if type(items) is not list or type(handles) is not list or not 1 <= len(items) <= 6 or len(items) != len(handles):
        raise BoundaryError("bounded one-to-one handle evidence required")
    ids, clean = [], []
    for h, item in zip(handles, items):
        if type(h) is not dict or type(item) is not dict or set(item) != {"node_id", "source", "sha256"}:
            raise BoundaryError("invalid evidence item")
        node, source = item["node_id"], item["source"]
        if type(node) is not str or not 0 < len(node) <= 512 or h.get("nodeId") != node:
            raise BoundaryError("handle identity mismatch")
        if type(source) is not str or not 0 < len(source) <= 2048:
            raise BoundaryError("invalid evidence source")
        checked_digest(item["sha256"])
        ids.append(node)
        clean.append(dict(item))
    if len(ids) != len(set(ids)) or nemo_digest(clean) != hit.get("evidence_set_sha256"):
        raise BoundaryError("duplicate handles or mismatched evidence digest")
    generation = checked_digest(hit.get("generation_sha256"))
    return {"items": clean, "handles": [{"nodeId": i} for i in ids],
            "evidence_set_sha256": nemo_digest(clean), "generation_sha256": generation}


@dataclass(frozen=True)
class Principal:
    principal_sha256: str
    tenant_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        for v in (self.principal_sha256, self.tenant_sha256, self.policy_sha256):
            checked_digest(v)


class EvidenceBoundController:
    """Proposal-only synthetic controller; PRE/POST gates enclose real inference.

    Callbacks are trusted in-process dependencies, not user-supplied request data.
    Construct through from_installed() for actual pinned estate integrations.
    Witness refusal/errors and edited histories never advance an accepted cache.
    """
    def __init__(self, model, principal: Principal, *, retrieve: Callable, witness: Callable,
                 authorize: Callable, formulas: dict) -> None:
        self._model = deepcopy(model).eval()
        self._principal = principal
        self._retrieve, self._witness, self._authorize = retrieve, witness, authorize
        self._formulas = deepcopy(formulas)
        self._weight_digest = digest(weight_bytes(self._model))
        self._cache = None
        self._cache_digest = None
        self._scope = None
        self._history = ()
        self._lock = threading.Lock()
        self.installation = {}

    @classmethod
    def from_installed(cls, model, principal: Principal, *, authorize: Callable):
        if any(os.environ.get(key) for key in ("SECOND_BRAIN_CORPUS", "AYLLU_BRAIN_CORPUS")):
            raise BoundaryError("only the packaged public corpus is admitted in this research adapter")
        installs = {name: verified_install(name) for name in PINS}
        from second_brain.retrieve import SecondBrainIndex
        from szl_nemo.envelope import evaluate_envelope
        from inference.validate_control_plane import load
        contract = load()
        source = contract["formula_binding"]
        formulas = {"locked_proven_ids": source["locked_proven_ids"],
                    "locked_proven_count": source["locked_proven_count"],
                    "formal_source_repository": source["formal_source"]["repository"],
                    "formal_source_commit": source["formal_source"]["commit"],
                    "kernel_source_repository": source["kernel_source"]["repository"],
                    "kernel_source_commit": source["kernel_source"]["commit"],
                    "f_id_to_callable_mapping": "UNKNOWN_NOT_ASSERTED",
                    "requested_formula_ids": [], "applications": [], "authorization_basis_ids": [],
                    "lambda": source["lambda"]}
        index = SecondBrainIndex()
        instance = cls(model, principal, retrieve=index.navigator_context, witness=evaluate_envelope,
                       authorize=authorize, formulas=formulas)
        instance.installation = installs
        return instance

    def _envelope(self, stage: str, bound: dict, scope: Scope, observation: dict) -> dict:
        p = self._principal
        return {"schema": "szl.nemo.inference-envelope.v1", "stage": stage,
                "witness_identity": {"artifact_kind": "SOFTWARE_KERNEL", "generative": False, "not_nemotron": True},
                "scope": {"principal_id_sha256": p.principal_sha256, "tenant_id_sha256": p.tenant_sha256,
                          "access_decision": "ALLOW", "policy_revision": "sha256:"+p.policy_sha256},
                "model": {"id": "SZL-RLT-SYNTHETIC-REGISTER-RESEARCH", "revision": "sha256:"+self._weight_digest,
                          "adapter_revision": "NONE", "tokenizer_revision": "sha256:"+scope.tokenizer_sha256,
                          "template_revision": "sha256:"+scope.template_sha256},
                "runtime": {"engine": "szl-rlt-cpu", "version": "0.2.0", "hardware_fingerprint": observation["identity_sha256"]},
                "evidence": {"content_access": "HANDLES_ONLY", "grounding_required": True,
                             "handles": bound["handles"], "items": bound["items"],
                             "evidence_set_sha256": bound["evidence_set_sha256"]},
                "formulas": self._formulas,
                "authority": {"model_authority": "PROPOSAL_ONLY", "executed": False, "execution_authority": "NONE"},
                "witness_history": [] if stage == "PRE_GENERATION" else ["PRE_GENERATION"],
                "claims": [{"label": "MEASURED", "statement_sha256": nemo_digest(observation)}],
                "tool_intent": None, "action_admission": None, "receipt": None,
                "tool_result": None, "postcondition": None,
                "continuity": observation}

    def _check(self, envelope: dict) -> dict:
        decision = self._witness(deepcopy(envelope))
        if getattr(decision, "input_hash", None) != "sha256:"+nemo_digest(envelope):
            raise BoundaryError("witness does not bind this exact envelope")
        if getattr(decision, "decision", None) != "ALLOW":
            raise BoundaryError("Nemo refused this inference stage")
        return {"stage": envelope["stage"], "decision": "ALLOW", "input_sha256": nemo_digest(envelope),
                "rule_version": str(decision.rule_version)}

    def evaluate(self, history: list[int], *, evidence_query: str) -> dict:
        if not self._lock.acquire(blocking=False):
            raise BoundaryError("controller already in use")
        try:
            if type(history) is not list or not 1 <= len(history) <= 64:
                raise BoundaryError("synthetic history must contain 1..64 operations")
            target = oracle(history)[-1]
            if type(evidence_query) is not str or not 1 <= len(evidence_query) <= 200:
                raise BoundaryError("bounded evidence query required")
            if self._authorize(self._principal, evidence_query) is not True:
                raise BoundaryError("controller authorization denied")
            bound = evidence_binding(self._retrieve(evidence_query, k=3))
            scope = Scope(tenant_sha256=digest(canonical([self._principal.tenant_sha256,self._principal.principal_sha256])),
                          policy_sha256=self._principal.policy_sha256,
                          evidence_sha256=nemo_digest(bound),
                          tokenizer_sha256=digest(b"SET0 SET1 FLIP KEEP REVOKE : 0 1 2 3 4"),
                          template_sha256=digest(b"synthetic-register-state-labels-8-9-10.v1"))
            # Restore into a disposable session; no accepted-state mutation on a refused POST.
            reuse = self._scope == scope and len(history)>len(self._history) and tuple(history[:len(self._history)])==self._history
            if reuse:
                candidate = Session.restore(self._model, scope, self._cache, expected_sha256=self._cache_digest,
                                            consumed_tokens=self._history)
                tokens = history[len(self._history):]
                mode = "COMPATIBLE_PREFIX_REUSED"
            else:
                candidate = Session(self._model, scope)
                tokens = history
                mode = "FRESH" if self._scope is None else "EVIDENCE_OR_HISTORY_REPLAY"
            pre = self._check(self._envelope("PRE_GENERATION", bound, scope, candidate.observation()))
            logits = candidate.append(tokens)
            pred = int(logits[0,-1,LABEL_OFFSET:LABEL_OFFSET+3].argmax())
            post_observation = candidate.observation()
            post_observation["output_sha256"] = digest(canonical({"predicted_register_state": pred}))
            post = self._check(self._envelope("POST_GENERATION", bound, scope, post_observation))
            artifact = candidate.checkpoint()
            self._cache, self._cache_digest = artifact, digest(artifact)
            self._scope, self._history = scope, tuple(history)
            status = "ABSTAIN_UNKNOWN" if target == 2 else "VERIFIED_SYNTHETIC" if pred == target else "ABSTAIN_MODEL_MISMATCH"
            return {"schema": "szl.rlt.estate-observation.v1", "status": status, "continuity_mode": mode,
                    "task_verified": status == "VERIFIED_SYNTHETIC", "verification_scope": "EXACT_SYNTHETIC_REGISTER_FINAL_STATE_ONLY",
                    "model_matched_oracle": pred == target, "evidence_count": len(bound["items"]),
                    "evidence_sha256": scope.evidence_sha256, "observation": post_observation,
                    "witness": [pre,post], "installation": self.installation,
                    "executed": False, "execution_authority": "NONE", "receipt_status": "UNSIGNED"}
        finally:
            self._lock.release()


def main() -> int:
    """Installed-package smoke using real public index and Nemo, no live endpoints."""
    import argparse
    from .training import load_candidate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.weights.stat().st_size > 1024*1024:
        raise BoundaryError("candidate resource limit")
    model = load_candidate(args.weights.read_bytes(), expected_sha256=args.expected_sha256)
    p = Principal(digest(b"synthetic-operator"),digest(b"public-research-tenant"),digest(b"read-only-synthetic-research"))
    def authorize(principal, query):
        return principal == p and query == "bounded terminating receipt closed"
    c = EvidenceBoundController.from_installed(model,p,authorize=authorize)
    first = c.evaluate([0,2,3,1],evidence_query="bounded terminating receipt closed")
    second = c.evaluate([0,2,3,1,3],evidence_query="bounded terminating receipt closed")
    edited = c.evaluate([1,2,3,1,3],evidence_query="bounded terminating receipt closed")
    assert second["continuity_mode"] == "COMPATIBLE_PREFIX_REUSED"
    assert edited["continuity_mode"] == "EVIDENCE_OR_HISTORY_REPLAY"
    print(json.dumps({"schema":"szl.rlt.installed-estate-smoke.v1","observations":[first,second,edited],
                      "publication_eligible":False,"production_qualified":False},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
