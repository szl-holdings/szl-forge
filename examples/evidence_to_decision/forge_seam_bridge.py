"""forge_seam_bridge.py - REFERENCE glue between the envelope adapter and Forge's
``inference/production.py`` protocol seams (Retriever / Hydrator / Generator / EnvelopeWitness).

Proposal-only.  No action authority.  Stdlib only.  Synthetic data only.  It does not
modify or monkeypatch production.py; it only supplies callables to ``production_infer``.

What the bridge adds, and nothing else:
  * ``retriever(query, k)``: Forge's Retriever signature over a caller-supplied candidate
    index; the adapter's ``Session.retrieve`` decides which candidates are admitted.
  * ``identity()``: the adapter's destination identity plus the model/runtime pins that
    Forge's generator-identity validation requires (supplied here as configuration).
  * ``__call__(context)``: calls the adapter's single-use ``Session.generator`` and shapes
    its text into Forge's generation record (output_schema, claims, citations).
  * ``build_request()``: principal and tenant are taken from the session's trusted
    principal, never from caller input.
  * ``LocalSpyWitness``: a local stand-in for the envelope witness.  It is NOT the real
    witness; it echoes the envelope digest and returns a scripted decision.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional, Sequence

from a11oy_forge_envelope_adapter import Chunk, Session, canonical_sha256, text_sha256

CITE_RE = re.compile(r"\[cite:([A-Za-z0-9._-]+)\]")
OUTPUT_SCHEMA = "szl.e2d.proposal-text/v0"


class ForgeSeams:
    """One request's Forge-shaped seams over one adapter Session."""

    def __init__(self, session: Session, candidate_index: Callable[[str], Sequence[Chunk]],
                 *, model_pins: Mapping[str, str], runtime: Mapping[str, str],
                 claim_label: str = "REPORTED"):
        self.session = session
        self._index = candidate_index
        self._model_pins = dict(model_pins)
        self._runtime = dict(runtime)
        self._label = claim_label
        self.retrieve_calls = 0

    def build_request(self, prompt: str, **extra: Any) -> dict:
        p = self.session.principal
        req = {"prompt": prompt, "principal_id": p.principal_id, "tenant_id": p.tenant_id,
               "policy_revision": self.session._a.policy_revision, "request_id": self.session.request_id}
        req.update(extra)
        return req

    # Forge Retriever
    def retriever(self, query: str, k: int = 6) -> dict:
        self.retrieve_calls += 1
        return self.session.retrieve(list(self._index(query))[:k])

    # Forge Hydrator
    def hydrator(self, handles, *, principal_id: str, tenant_id: str, policy_revision: str) -> dict:
        return self.session.hydrator(handles, principal_id=principal_id, tenant_id=tenant_id,
                                     policy_revision=policy_revision)

    # Forge Generator
    def identity(self) -> dict:
        base = self.session.identity()
        model = dict(base["model"])
        model.update(self._model_pins)
        return {"model": model, "runtime": dict(self._runtime)}

    def __call__(self, context: Mapping[str, Any]) -> dict:
        out = self.session.generator(context)          # adapter re-checks rights at egress
        text = out["text"]
        nodes = {str(d.get("node_id")) for d in context.get("evidence", [])}
        cites = []
        for c in CITE_RE.findall(text):
            if c not in cites:
                cites.append(c)
        statement = CITE_RE.sub("", text).strip()
        return {"text": text, "output_schema": OUTPUT_SCHEMA,
                "claims": [{"label": self._label, "statement_sha256": text_sha256(statement),
                            "supporting_node_ids": [c for c in cites if c in nodes]}],
                "citations": cites, "metrics": {}}


class LocalSpyWitness:
    """Local stand-in for the envelope witness: records stages, never authorizes anything."""

    def __init__(self, decision: str = "ALLOW", on_stage: Optional[Mapping[str, Callable[[], None]]] = None):
        self.decision = decision
        self.on_stage = dict(on_stage or {})
        self.stages: list = []

    def __call__(self, envelope: Mapping[str, Any]) -> dict:
        stage = str(envelope.get("stage"))
        self.stages.append(stage)
        hook = self.on_stage.get(stage)
        if hook is not None:
            hook()
        return {"decision": self.decision, "rule_version": "local-spy-witness/0",
                "input_hash": "sha256:" + canonical_sha256(dict(envelope)),
                "violated_rules": [] if self.decision == "ALLOW" else ["SPY_SCRIPTED"], "reasons": []}
