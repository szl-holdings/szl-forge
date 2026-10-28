#!/usr/bin/env python3
"""a11oy_forge_envelope_adapter.py -- REFERENCE adapter: A11oy generation/chunk -> Forge envelope.

Stdlib only, synthetic data only. This is NOT a competing controller. It wraps the
three Forge protocol seams (szl-forge 35f34c05 inference/production.py L54-76):

    Retriever.__call__(query, k)                                   -> handles only
    Hydrator.__call__(handles, *, principal_id, tenant_id, policy_revision)
    Generator.identity() / Generator.__call__(context)

so that every seam is bound to a TRUSTED principal and re-checks source rights:
  * at retrieval     (purpose "read")
  * before hydration (purpose "hydrate")
  * immediately before the single fixed-destination model egress (purpose "egress").

Anything wrong -- wrong tenant, revoked grant, expired grant, stale policy epoch,
stale generation, revision/digest mismatch, unknown source/chunk/purpose, registry
unavailable -- raises Refusal BEFORE any model call. The backend is called at most
once per session and there is no provider fallback.

Authority fields (human_approval, signature_status, grant ids, principal/tenant,
policy epoch, owner, visibility ...) are never read from the untrusted request: a
request carrying any of them is refused. Outputs are PROPOSAL_ONLY and UNSIGNED and
human review stays PENDING. NOTE: every "digest" check here is a shape/equality
check on hex strings; it is NOT cryptographic authority (no signature is verified
and none is produced).

The rights registry is coded to a minimal Protocol (resolve_subject / source_grant /
admit_transfer / is_revoked / current_policy_epoch / active_generation).  It is a
PROPOSED interface; no upstream registry signature is asserted here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

VERSION = "0.1.0-reference"
PURPOSES = frozenset({"read", "hydrate", "egress"})
VISIBILITIES = frozenset({"PUBLIC", "PRIVATE"})
STORE_KINDS = frozenset({"PUBLIC_DEFAULT", "SCOPED_PRIVATE"})
TRUST_DOMAINS = frozenset({"INTERNAL", "EXTERNAL"})
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ALLOWED_REQUEST_KEYS = frozenset({"prompt", "k", "request_id"})
AUTHORITY_FIELDS = frozenset({
    "human_approval", "human_review", "approved", "approved_by", "approval",
    "signature", "signature_status", "signed", "verified",
    "grant_id", "grant_ids", "grants", "principal_id", "principal", "tenant_id",
    "tenant", "policy_epoch", "policy_revision", "owner", "visibility", "purposes",
    "revoked", "expires_at", "authority", "destination", "provider", "model",
})
AUTHORITY_NOTE = ("digest/label checks are shape+equality checks only; "
                  "they are not cryptographic authority")


class Refusal(Exception):
    """Fail-closed refusal with a stable reason code."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def canonical_bytes(value: Any) -> bytes:  # identical to Forge production.py L86-92
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- data shapes
@dataclass(frozen=True)
class TrustedPrincipal:
    principal_id: str
    tenant_id: str
    minted_by: str          # who minted it (controller-side), never the request


@dataclass(frozen=True)
class SourceGrant:
    grant_id: str
    source_id: str
    principal_id: str
    tenant_id: str
    owner: str
    visibility: str                 # PUBLIC | PRIVATE
    source_revision: str            # 40-hex immutable commit
    source_digest: str              # 64-hex full digest of the pinned source
    purposes: frozenset
    policy_epoch: int
    expires_at: int                 # epoch seconds (controller clock)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    generation_id: str
    source_revision: str
    source_digest: str
    body_sha256: str


@dataclass(frozen=True)
class StoreProfile:
    store_id: str
    kind: str                       # PUBLIC_DEFAULT | SCOPED_PRIVATE


@dataclass(frozen=True)
class FixedDestination:
    provider: str
    model_id: str
    model_revision: str
    trust_domain: str               # INTERNAL | EXTERNAL
    endpoint_id: str

    def identity(self) -> dict:
        return {"model": {"id": self.model_id, "revision": self.model_revision},
                "runtime": {"provider": self.provider, "endpoint_id": self.endpoint_id,
                            "trust_domain": self.trust_domain}}


class RightsRegistry(Protocol):
    """PROPOSED rights-registry interface; no upstream signature is asserted."""

    def resolve_subject(self, credential: Any) -> Optional[TrustedPrincipal]: ...
    def source_grant(self, principal: TrustedPrincipal, source_id: str) -> Optional[SourceGrant]: ...
    def admit_transfer(self, principal: TrustedPrincipal, grant: SourceGrant,
                       destination: FixedDestination) -> bool: ...
    def is_revoked(self, grant_id: str) -> bool: ...
    def current_policy_epoch(self) -> int: ...
    def active_generation(self, source_id: str) -> Optional[str]: ...


# ------------------------------------------------------------- read-only DAG
class EvidenceDAG:
    """Append-only, hash-linked provenance DAG. Hashes only; no bodies."""

    KINDS = ("request", "chunk", "hydration_decision", "model_call", "output",
             "human_review", "refusal")

    def __init__(self) -> None:
        self._nodes: list[dict] = []

    def add(self, kind: str, parents: Sequence[str], **fields: Any) -> str:
        if kind not in self.KINDS:
            raise Refusal("DAG_KIND_UNKNOWN", kind)
        known = {n["id"] for n in self._nodes}
        if any(p not in known for p in parents):
            raise Refusal("DAG_PARENT_UNKNOWN")
        for k, v in fields.items():
            if k in ("body", "content", "text", "prompt"):
                raise Refusal("DAG_BODY_FORBIDDEN", k)
        node = {"kind": kind, "parents": sorted(parents), **fields}
        node["id"] = canonical_sha256(node)
        self._nodes.append(node)
        return node["id"]

    def export(self) -> list:
        return copy.deepcopy(self._nodes)


def verify_dag(nodes: Sequence[Mapping]) -> bool:
    seen: set = set()
    for n in nodes:
        body = {k: v for k, v in n.items() if k != "id"}
        if canonical_sha256(body) != n.get("id"):
            return False
        if any(p not in seen for p in n.get("parents", [])):
            return False
        seen.add(n["id"])
    return True


# ---------------------------------------------------------------- the adapter
class EnvelopeAdapter:
    """Factory: one Session per request. Holds only controller-side config."""

    def __init__(self, registry: RightsRegistry, store: StoreProfile,
                 destination: FixedDestination, backend: Callable[[Mapping], Mapping],
                 content_store: Mapping[str, str], clock: Callable[[], int],
                 policy_revision: str):
        if store.kind not in STORE_KINDS:
            raise Refusal("STORE_KIND_UNKNOWN")
        if destination.trust_domain not in TRUST_DOMAINS:
            raise Refusal("DESTINATION_UNKNOWN")
        if not HEX40.match(policy_revision or ""):
            raise Refusal("POLICY_REVISION_NOT_PINNED")
        self.registry = registry
        self.store = store
        self.destination = destination
        self._backend = backend
        self._content = content_store
        self._clock = clock
        self.policy_revision = policy_revision

    def open(self, credential: Any, request: Mapping) -> "Session":
        if not isinstance(request, Mapping):
            raise Refusal("REQUEST_MALFORMED")
        smuggled = sorted(set(request) & AUTHORITY_FIELDS)
        if smuggled:
            raise Refusal("UNTRUSTED_AUTHORITY_FIELD", ",".join(smuggled))
        unknown = sorted(set(request) - ALLOWED_REQUEST_KEYS)
        if unknown:
            raise Refusal("REQUEST_FIELD_UNKNOWN", ",".join(unknown))
        try:
            principal = self.registry.resolve_subject(credential)
        except Exception as exc:  # registry unavailable -> fail closed
            raise Refusal("REGISTRY_UNAVAILABLE", type(exc).__name__)
        if not isinstance(principal, TrustedPrincipal):
            raise Refusal("PRINCIPAL_UNKNOWN")
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise Refusal("PROMPT_REQUIRED")
        return Session(self, principal, dict(request))


class Session:
    """Exposes Forge-shaped retriever/hydrator/generator callables for ONE request."""

    def __init__(self, adapter: EnvelopeAdapter, principal: TrustedPrincipal, request: dict):
        self._a = adapter
        self.principal = principal
        self._prompt = request["prompt"]
        self.request_id = str(request.get("request_id") or
                              text_sha256(principal.tenant_id + "\0" +
                                          principal.principal_id + "\0" + self._prompt)[:24])
        self.dag = EvidenceDAG()
        self._req_node = self.dag.add(
            "request", [], request_id=self.request_id,
            principal_id_sha256=text_sha256(principal.principal_id),
            tenant_id_sha256=text_sha256(principal.tenant_id),
            prompt_sha256=text_sha256(self._prompt),
            policy_revision=adapter.policy_revision)
        self._admitted: dict[str, tuple] = {}     # chunk_id -> (Chunk, SourceGrant, dag id)
        self._hydrated: Optional[dict] = None     # node_id -> sha256
        self._hyd_node: Optional[str] = None
        self._egress_used = False
        self.state = "OPEN"

    # -- the single rights check used at every seam -------------------------
    def _check(self, chunk: Chunk, purpose: str) -> SourceGrant:
        reg = self._a.registry
        if purpose not in PURPOSES:
            raise Refusal("PURPOSE_UNKNOWN", str(purpose))
        try:
            grant = reg.source_grant(self.principal, chunk.source_id)
            epoch = reg.current_policy_epoch()
            active = reg.active_generation(chunk.source_id)
            revoked = None if grant is None else reg.is_revoked(grant.grant_id)
        except Exception as exc:
            raise Refusal("REGISTRY_UNAVAILABLE", type(exc).__name__)
        if not isinstance(grant, SourceGrant):
            raise Refusal("GRANT_UNKNOWN")
        if grant.source_id != chunk.source_id:
            raise Refusal("GRANT_SOURCE_MISMATCH")
        if grant.principal_id != self.principal.principal_id:
            raise Refusal("PRINCIPAL_MISMATCH")
        if grant.tenant_id != self.principal.tenant_id:
            raise Refusal("TENANT_MISMATCH")
        if revoked is not False:
            raise Refusal("GRANT_REVOKED")
        if not isinstance(grant.expires_at, int) or self._a._clock() >= grant.expires_at:
            raise Refusal("GRANT_EXPIRED")
        if not isinstance(epoch, int) or grant.policy_epoch != epoch:
            raise Refusal("POLICY_EPOCH_STALE")
        if purpose not in grant.purposes:
            raise Refusal("PURPOSE_NOT_GRANTED", purpose)
        if grant.visibility not in VISIBILITIES:
            raise Refusal("VISIBILITY_UNKNOWN")
        if grant.visibility == "PRIVATE" and self._a.store.kind != "SCOPED_PRIVATE":
            raise Refusal("PRIVATE_SOURCE_IN_PUBLIC_STORE")
        if active is None or chunk.generation_id != active:
            raise Refusal("GENERATION_STALE")
        if not HEX40.match(chunk.source_revision) or chunk.source_revision != grant.source_revision:
            raise Refusal("SOURCE_REVISION_MISMATCH")
        if not HEX64.match(chunk.source_digest) or chunk.source_digest != grant.source_digest:
            raise Refusal("SOURCE_DIGEST_MISMATCH")
        return grant

    def _refuse(self, code: str, parents: Sequence[str], **f: Any) -> None:
        self.dag.add("refusal", parents, code=code, **f)
        self.state = "REFUSED"
        raise Refusal(code)

    # -- Retriever seam ------------------------------------------------------
    def retrieve(self, candidates: Sequence[Chunk]) -> dict:
        """Filter retriever candidates (purpose=read). Denied ones -> hashes only."""
        if self.state != "OPEN":
            raise Refusal("SESSION_STATE", self.state)
        handles, evidence = [], []
        for c in candidates:
            if not isinstance(c, Chunk):
                raise Refusal("CHUNK_MALFORMED")
            try:
                grant = self._check(c, "read")
            except Refusal as r:
                self.dag.add("chunk", [self._req_node], decision="DENIED", code=r.code,
                             chunk_id_sha256=text_sha256(c.chunk_id))
                continue
            nid = self.dag.add("chunk", [self._req_node], decision="ADMITTED_READ",
                               chunk_id_sha256=text_sha256(c.chunk_id),
                               source_revision=c.source_revision,
                               source_digest=c.source_digest,
                               generation_id=c.generation_id, body_sha256=c.body_sha256,
                               grant_id_sha256=text_sha256(grant.grant_id))
            self._admitted[c.chunk_id] = (c, grant, nid)
            handles.append({"nodeId": c.chunk_id, "nodeKind": "CHUNK", "label": "GRANTED",
                            "note": ""})
            evidence.append({"node_id": c.chunk_id, "source": c.source_id,
                             "sha256": c.body_sha256})
        self.state = "RETRIEVED"
        return {"content_access": "HANDLES_ONLY", "handles": handles, "evidence": evidence,
                "ready": bool(handles), "evidence_set_sha256": canonical_sha256(evidence),
                "ranking_receipt": {}}

    # -- Hydrator seam (Forge Hydrator signature) ----------------------------
    def hydrator(self, handles: Sequence[Mapping], *, principal_id: str, tenant_id: str,
                 policy_revision: str) -> dict:
        if self.state != "RETRIEVED":
            raise Refusal("SESSION_STATE", self.state)
        if principal_id != self.principal.principal_id or tenant_id != self.principal.tenant_id:
            self._refuse("HYDRATION_PRINCIPAL_MISMATCH", [self._req_node])
        if policy_revision != self._a.policy_revision:
            self._refuse("HYDRATION_POLICY_MISMATCH", [self._req_node])
        ids = [str(h.get("nodeId", "")) for h in handles]
        if not ids or len(ids) != len(set(ids)):
            self._refuse("HYDRATION_HANDLES_INVALID", [self._req_node])
        docs, evidence, parents = [], [], []
        for cid in ids:
            if cid not in self._admitted:
                self._refuse("CHUNK_UNKNOWN", [self._req_node],
                             chunk_id_sha256=text_sha256(cid))
            chunk, _, nid = self._admitted[cid]
            parents.append(nid)
            try:
                self._check(chunk, "hydrate")               # RECHECK before any body read
            except Refusal as r:
                self._refuse(r.code, [nid], stage="hydrate")
            body = self._a._content.get(cid)
            if not isinstance(body, str) or text_sha256(body) != chunk.body_sha256:
                self._refuse("BODY_DIGEST_MISMATCH", [nid])
            docs.append({"node_id": cid, "source": chunk.source_id,
                         "sha256": chunk.body_sha256, "content": body})
            evidence.append({"node_id": cid, "source": chunk.source_id,
                             "sha256": chunk.body_sha256})
        ev_digest = canonical_sha256(evidence)
        self._hyd_node = self.dag.add("hydration_decision", parents, decision="AUTHORIZED",
                                      evidence_set_sha256=ev_digest)
        self._hydrated = {d["node_id"]: d["sha256"] for d in docs}
        self.state = "HYDRATED"
        return {"content_access": "CONTROLLER_ONLY", "state": "AUTHORIZED_CONTENT_READY",
                "principal_id_sha256": text_sha256(principal_id),
                "tenant_id_sha256": text_sha256(tenant_id),
                "policy_revision": policy_revision, "evidence_set_sha256": ev_digest,
                "documents": docs, "ready": bool(docs),
                "raw_graph_nodes_admitted_to_gradients": 0}

    # -- Generator seam ------------------------------------------------------
    def identity(self) -> dict:
        return self._a.destination.identity()

    def generator(self, context: Mapping) -> dict:
        if self.state != "HYDRATED" or self._hydrated is None:
            raise Refusal("SESSION_STATE", self.state)
        if self._egress_used:
            self._refuse("EGRESS_ALREADY_USED", [self._hyd_node])
        if not isinstance(context, Mapping) or context.get("request_id") != self.request_id:
            self._refuse("CONTEXT_REQUEST_MISMATCH", [self._hyd_node])
        ev = context.get("evidence")
        if not isinstance(ev, list):
            self._refuse("CONTEXT_EVIDENCE_INVALID", [self._hyd_node])
        got = {}
        for d in ev:
            if not isinstance(d, Mapping):
                self._refuse("CONTEXT_EVIDENCE_INVALID", [self._hyd_node])
            got[str(d.get("node_id"))] = (str(d.get("sha256")), str(d.get("content", "")))
        if set(got) != set(self._hydrated):
            self._refuse("CONTEXT_EVIDENCE_SET_MISMATCH", [self._hyd_node])
        for nid, (dg, content) in got.items():
            if dg != self._hydrated[nid] or text_sha256(content) != dg:
                self._refuse("CONTEXT_DIGEST_MISMATCH", [self._hyd_node])
        dest = self._a.destination
        for nid in self._hydrated:
            chunk = self._admitted[nid][0]
            try:
                grant = self._check(chunk, "egress")        # RECHECK immediately before egress
            except Refusal as r:
                self._refuse(r.code, [self._hyd_node], stage="egress")
            if grant.visibility == "PRIVATE" and dest.trust_domain != "INTERNAL":
                self._refuse("PRIVATE_TO_EXTERNAL", [self._hyd_node])
            try:
                ok = self._a.registry.admit_transfer(self.principal, grant, dest)
            except Exception:
                ok = None
            if ok is not True:
                self._refuse("TRANSFER_NOT_ADMITTED", [self._hyd_node])
        call_node = self.dag.add("model_call", [self._hyd_node],
                                 destination_sha256=canonical_sha256(dest.identity()),
                                 context_sha256=canonical_sha256(dict(context)))
        self._egress_used = True                            # single use, set before the call
        try:
            raw = self._a._backend(dict(context))           # the ONE fixed destination
        except Exception as exc:                            # no fallback provider
            self._refuse("BACKEND_FAILED_NO_FALLBACK", [call_node], error=type(exc).__name__)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("text"), str):
            self._refuse("BACKEND_OUTPUT_MALFORMED", [call_node])
        out_digest = text_sha256(raw["text"])
        out_node = self.dag.add("output", [call_node], output_sha256=out_digest)
        self.dag.add("human_review", [out_node], state="PENDING")
        self.state = "PROPOSED"
        # authority fields are set HERE, never copied from backend or request
        return {"text": raw["text"], "output_sha256": out_digest,
                "authority": "PROPOSAL_ONLY", "signature_status": "UNSIGNED",
                "human_review": "PENDING", "human_approval": None,
                "authority_note": AUTHORITY_NOTE}


# ---------------------------------------------------- synthetic reference registry
class InMemoryRegistry:
    """Synthetic test registry. A real deployment supplies its own registry."""

    def __init__(self, epoch: int = 1):
        self.principals: dict = {}
        self.grants: dict = {}            # (principal_id, source_id) -> SourceGrant
        self.revoked: set = set()
        self.epoch = epoch
        self.generations: dict = {}
        self.transfers: set = set()       # (grant_id, endpoint_id)
        self.available = True

    def _up(self) -> None:
        if not self.available:
            raise ConnectionError("registry unavailable")

    def resolve_subject(self, credential):
        self._up()
        return self.principals.get(credential)

    def source_grant(self, principal, source_id):
        self._up()
        return self.grants.get((principal.principal_id, source_id))

    def admit_transfer(self, principal, grant, destination):
        self._up()
        return (grant.grant_id, destination.endpoint_id) in self.transfers

    def is_revoked(self, grant_id):
        self._up()
        return grant_id in self.revoked

    def current_policy_epoch(self):
        self._up()
        return self.epoch

    def active_generation(self, source_id):
        self._up()
        return self.generations.get(source_id)
