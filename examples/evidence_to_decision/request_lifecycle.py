"""request_lifecycle.py - stdlib reference model of a governed synthesis request.

Contract 4 (request lifecycle, revocation and cancellation).  REFERENCE /
ACCEPTANCE model on synthetic data only.  It is not a11oy code, it does not
reproduce any race in a11oy, and it performs no network or file I/O.

States
    ADMITTED -> HYDRATED -> COMMITTED_TO_DISPATCH -> DISPATCHED -> STREAMING
    -> COMPLETED, with DENIED / CANCELLED / FAILED / UNCERTAIN branches.

Linearization point
    ``Controller.commit_lock`` (one threading.Lock) serialises (a) each
    request's COMMIT critical section -- the final epoch / generation /
    destination / deadline / cancel check plus the ``egress_release`` record
    that hands the payload to the dispatch worker -- and (b) every policy
    mutation (revocation of a source digest, principal or destination, which
    increments ``policy.epoch``).  The ``seq`` order of those records in
    ``Controller.ledger`` IS the linearization order.  A commit linearized
    before a revocation is listed in that revocation's
    ``disclosed_before_revocation``; a commit attempted after it is DENIED.

    The provider send is OUTSIDE the lock and is NOT atomic with the check.
    After COMMIT the worker sends without further policy checks, and the
    payload is accounted as disclosed-possible from the commit record onward.

Guards that the mutation sweep removes are tagged ``# GUARD:<id>`` / ``# LOCK:<id>``.
"""
from __future__ import annotations

import contextlib  # noqa: F401  (used by lock mutants in the mutation sweep)
import hashlib
import itertools
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, Optional


class State(str, Enum):
    ADMITTED = "ADMITTED"
    HYDRATED = "HYDRATED"
    COMMITTED = "COMMITTED_TO_DISPATCH"
    DISPATCHED = "DISPATCHED"
    STREAMING = "STREAMING"
    COMPLETED = "COMPLETED"
    DENIED = "DENIED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


TERMINAL = frozenset({State.COMPLETED, State.DENIED, State.CANCELLED, State.FAILED})
# UNCERTAIN is quiescent (no further effects) but resolvable: to CANCELLED when
# worker termination is observed, or to DISPATCHED through a verified-idempotent
# re-send with the same key.  It is never silently retried.

ALLOWED_TRANSITIONS = {
    State.ADMITTED: {State.HYDRATED, State.DENIED, State.CANCELLED, State.FAILED},
    State.HYDRATED: {State.COMMITTED, State.DENIED, State.CANCELLED, State.FAILED},
    State.COMMITTED: {State.DISPATCHED, State.FAILED, State.UNCERTAIN},
    State.DISPATCHED: {State.STREAMING, State.CANCELLED, State.FAILED, State.UNCERTAIN},
    State.STREAMING: {State.COMPLETED, State.CANCELLED, State.FAILED, State.UNCERTAIN},
    State.UNCERTAIN: {State.CANCELLED, State.DISPATCHED},
    State.COMPLETED: set(), State.DENIED: set(), State.CANCELLED: set(), State.FAILED: set(),
}

# Effect kind -> states in which the effect may occur.  Audit records
# (transitions, receipts, revocations) are decisions, not effects.
EFFECTS_ALLOWED_IN_STATE = {
    "source_read": {State.ADMITTED},              # internal; only after authorization
    "egress_release": {State.HYDRATED},           # only inside the commit critical section
    "provider_send": {State.COMMITTED, State.UNCERTAIN},  # attempt; UNCERTAIN only for idempotent resend
    "stream_out": {State.DISPATCHED, State.STREAMING},
    "cancel_request": {State.DISPATCHED, State.STREAMING},
    "cache_write": {State.STREAMING},             # only inside the completion commit
    "history_write": {State.STREAMING},
}
EXTERNAL_EFFECTS = frozenset(k for k in EFFECTS_ALLOWED_IN_STATE if k != "source_read") | {"derived_write"}


class LifecycleViolation(AssertionError):
    """An effect or transition the state machine does not allow."""


class RetryRefused(RuntimeError):
    pass


class ProviderNotSent(RuntimeError):
    """Connection refused before any request byte left the controller."""


class ProviderOutcomeUnknown(RuntimeError):
    """Bytes may have been delivered; no acknowledgement was observed."""


class ProviderFailed(RuntimeError):
    """Explicit provider error after acknowledgement."""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass(frozen=True)
class Destination:
    provider_id: str
    model_id: str
    trust_domain: str
    endpoint: str


@dataclass(frozen=True)
class SourceRef:
    generation_id: str
    chunk_id: str
    sha256: str


class GenerationStore:
    """Immutable generations with one active pointer (rollback = re-point)."""

    def __init__(self):
        self.generations: dict[str, dict[str, str]] = {}
        self.active: Optional[str] = None
        self.reads = 0

    def publish(self, generation_id: str, chunks: dict, activate: bool = True) -> None:
        self.generations[generation_id] = dict(chunks)
        if activate:
            self.active = generation_id

    def rollback(self, generation_id: str) -> None:
        if generation_id not in self.generations:
            raise KeyError(generation_id)
        self.active = generation_id

    def ref(self, generation_id: str, chunk_id: str) -> SourceRef:
        return SourceRef(generation_id, chunk_id, sha(self.generations[generation_id][chunk_id]))

    def read(self, ref: SourceRef) -> str:
        self.reads += 1
        return self.generations[ref.generation_id][ref.chunk_id]


@dataclass
class Policy:
    epoch: int = 1
    revoked_digests: set = field(default_factory=set)
    revoked_principals: set = field(default_factory=set)
    admitted_destinations: set = field(default_factory=set)
    admission_available: bool = True


class FakeProvider:
    """Deterministic provider. ``mode``: ok | down | lost | fail_after_ack.

    ``during_wait`` (optional callable) runs inside next_token, modelling an
    event that lands while the controller is waiting on the provider.
    """

    def __init__(self, tokens=("alpha ", "beta ", "gamma"), *, mode: str = "ok",
                 supports_cancel: bool = True, idempotency_verified: bool = False):
        self.tokens = list(tokens)
        self.mode = mode
        self.supports_cancel = supports_cancel
        self.idempotency_verified = idempotency_verified
        self.received: list = []          # (request_id, idempotency_key, payload_sha256)
        self.cancel_calls: list = []
        self.cursor: dict = {}
        self.finished: set = set()
        self.during_wait: Optional[Callable[[], None]] = None
        self._pid = itertools.count(1)
        self._by_key: dict = {}

    def send(self, request_id: str, idem_key: str, payload: str) -> str:
        if self.mode == "down":
            raise ProviderNotSent("connect refused")
        if self.idempotency_verified and idem_key in self._by_key:
            return self._by_key[idem_key]       # de-duplicated by the provider
        self.received.append((request_id, idem_key, sha(payload)))
        pid = f"p{next(self._pid)}"
        self._by_key[idem_key] = pid
        self.cursor[pid] = 0
        if self.mode == "lost":
            raise ProviderOutcomeUnknown("connection lost before acknowledgement")
        return pid

    def next_token(self, pid: str) -> Optional[str]:
        if self.during_wait is not None:
            fn, self.during_wait = self.during_wait, None
            fn()
        if self.mode in ("fail_after_ack", "down"):
            raise ProviderFailed("provider error after acknowledgement")
        i = self.cursor[pid]
        if i >= len(self.tokens):
            self.finished.add(pid)
            return None
        self.cursor[pid] = i + 1
        return self.tokens[i]

    def cancel(self, pid: Optional[str]) -> bool:
        self.cancel_calls.append(str(pid))
        if pid is not None and self.supports_cancel:
            self.finished.add(pid)
            return True
        return False


# --------------------------------------------------------------------------- #
# Request and controller
# --------------------------------------------------------------------------- #
@dataclass
class Request:
    request_id: str
    principal: str
    authenticated: bool
    sources: tuple
    destination: Destination
    deadline: float
    bound_epoch: int
    bound_generation: str
    idempotency_key: str
    state: State = State.ADMITTED
    reason: Optional[str] = None
    authorized: bool = False
    payload: Optional[str] = None
    payload_sha256: Optional[str] = None
    cancel_cause: Optional[str] = None
    cancel_requested_to_provider: bool = False
    cancel_confirmed: bool = False
    confirmed_via: Optional[str] = None
    sent: Optional[bool] = None
    provider_request_id: Optional[str] = None
    forwarded: list = field(default_factory=list)
    dropped_tokens: int = 0
    late_callbacks_dropped: int = 0
    client_connected: bool = True
    revoked_after_commit: bool = False
    commit_seq: Optional[int] = None

    @property
    def digests(self) -> frozenset:
        return frozenset(s.sha256 for s in self.sources)


class Controller:
    def __init__(self, store: GenerationStore, policy: Policy, provider: FakeProvider,
                 destination: Destination, clock: Optional[Clock] = None):
        self.store, self.policy, self.provider = store, policy, provider
        self.fixed_destination = destination
        self.clock = clock or Clock()
        self.commit_lock = threading.Lock()
        self._seq = itertools.count(1)
        self._ledger_lock = threading.Lock()
        self.ledger: list = []
        self.cache: dict = {}
        self.history: list = []
        self.derived: dict = {}
        self.inflight: dict = {}
        self.commit_hook: Optional[Callable[[Request], None]] = None
        self._ids = itertools.count(1)

    # -- ledger ------------------------------------------------------------- #
    def _append(self, kind: str, request_id: Optional[str], **data) -> dict:
        with self._ledger_lock:
            rec = {"seq": next(self._seq), "kind": kind, "request_id": request_id, **data}
            self.ledger.append(rec)
            return rec

    def _effect(self, req: Request, kind: str, **data) -> dict:
        if req.state not in EFFECTS_ALLOWED_IN_STATE[kind] or (kind == "source_read" and not req.authorized):
            raise LifecycleViolation(f"{kind} not allowed in {req.state.value}")
        return self._append("effect:" + kind, req.request_id, **data)

    def _audit(self, req: Optional[Request], event: str, **data) -> dict:
        return self._append("audit:" + event, req.request_id if req else None, **data)

    def effects(self, request_id=None, kind=None, external_only=False) -> list:
        out = []
        for r in self.ledger:
            if not r["kind"].startswith("effect:"):
                continue
            k = r["kind"][7:]
            if request_id is not None and r["request_id"] != request_id:
                continue
            if kind is not None and k != kind:
                continue
            if external_only and k not in EXTERNAL_EFFECTS:
                continue
            out.append(r)
        return out

    def _move(self, req: Request, to: State, reason: Optional[str] = None) -> None:
        if to not in ALLOWED_TRANSITIONS[req.state]:
            raise LifecycleViolation(f"{req.state.value} -> {to.value}")
        frm, req.state = req.state, to
        if reason is not None:
            req.reason = reason
        self._audit(req, "transition", frm=frm.value, to=to.value, reason=reason)
        if to in TERMINAL:
            self.inflight.pop(req.request_id, None)

    # -- admission ---------------------------------------------------------- #
    def admit(self, principal: str, sources, *, authenticated: bool = True,
              destination: Optional[Destination] = None, deadline: float = 100.0) -> Request:
        rid = f"r{next(self._ids)}"
        gens = {s.generation_id for s in sources}
        req = Request(rid, principal, authenticated, tuple(sources),
                      destination or self.fixed_destination, deadline, self.policy.epoch,
                      next(iter(gens)) if len(gens) == 1 else "", idempotency_key=sha("idem:" + rid))
        self.inflight[rid] = req
        self._audit(req, "admitted", epoch=req.bound_epoch, generation=req.bound_generation)
        return req

    # -- external events ---------------------------------------------------- #
    def _invalidate_descendants(self, digest: str) -> list:
        hit = []
        for store in (self.cache, self.derived):
            for key, entry in store.items():
                if digest in entry["provenance"] and entry["valid"]:
                    entry["valid"] = False                                      # GUARD:G20 descendant invalidation
                    hit.append(key)
        for entry in self.history:
            if digest in entry["provenance"] and entry["valid"]:
                entry["valid"] = False                                          # GUARD:G20h history invalidation
                hit.append(entry["request_id"])
        return hit

    def _revoke(self, what: str, value, match: Callable[[Request], bool]) -> dict:
        with self.commit_lock:                                                  # LOCK:L2 revoke side
            self.policy.epoch += 1
            disclosed = []
            for r in list(self.inflight.values()):
                if r.commit_seq is not None and match(r):
                    r.revoked_after_commit = True
                    disclosed.append(r.request_id)
            invalid = self._invalidate_descendants(value) if what == "source" else []
            return self._audit(None, "revocation", what=what, value=str(value),
                               epoch=self.policy.epoch,
                               disclosed_before_revocation=sorted(disclosed), invalidated=invalid)

    def revoke_source(self, digest: str) -> dict:
        self.policy.revoked_digests.add(digest)
        return self._revoke("source", digest, lambda r: digest in r.digests)

    def revoke_principal(self, principal: str) -> dict:
        self.policy.revoked_principals.add(principal)
        return self._revoke("principal", principal, lambda r: r.principal == principal)

    def revoke_destination(self, dest: Destination) -> dict:
        self.policy.admitted_destinations.discard(dest)
        return self._revoke("destination", dest, lambda r: r.destination == dest)

    def disconnect(self, req: Request) -> None:
        req.client_connected = False
        if req.cancel_cause is None:
            req.cancel_cause = "CLIENT_DISCONNECT"
        self._audit(req, "cancel_requested", cause="CLIENT_DISCONNECT")

    # -- lifecycle (generator: yields a point name BEFORE each step) --------- #
    def lifecycle(self, req: Request) -> Iterator[str]:
        yield "authorize"
        if not self._authorize(req):
            return
        yield "hydrate"
        if not self._hydrate(req):
            return
        yield "commit"
        if not self._commit(req):
            return
        yield "dispatch"
        if not self._dispatch(req):
            return
        yield "first_token"
        while True:
            more = self._stream_one(req)
            if more is False:
                return
            if more is None:
                break
            yield "token"
        yield "cache_commit"
        if not self._completion_commit(req):
            return
        yield "receipt_commit"
        self._receipt(req)

    def run(self, req: Request) -> Request:
        for _ in self.lifecycle(req):
            pass
        return req

    # -- pre-commit steps --------------------------------------------------- #
    def _pre_commit_stop(self, req: Request) -> bool:
        if req.cancel_cause is not None:                                        # GUARD:G08 cancel before commit
            req.cancel_confirmed, req.confirmed_via, req.sent = True, "NOT_SENT", False
            self._move(req, State.CANCELLED, req.cancel_cause)
            return True
        if self.clock.now() >= req.deadline:                                    # GUARD:G09 deadline before commit
            req.sent = False
            self._move(req, State.FAILED, "DEADLINE_EXPIRED")
            return True
        return False

    def _deny(self, req: Request, reason: str) -> bool:
        self._move(req, State.DENIED, reason)
        return False

    def _authorize(self, req: Request) -> bool:
        if self._pre_commit_stop(req):
            return False
        p = self.policy
        if not req.authenticated:                                               # GUARD:G01 authenticated subject
            return self._deny(req, "SUBJECT_UNAUTHENTICATED")
        if req.principal in p.revoked_principals:                               # GUARD:G21 principal revocation
            return self._deny(req, "PRINCIPAL_REVOKED")
        if not p.admission_available:                                           # GUARD:G03a admission service
            return self._deny(req, "PROVIDER_ADMISSION_UNAVAILABLE")
        if req.destination != self.fixed_destination or req.destination not in p.admitted_destinations:  # GUARD:G03 fixed destination
            return self._deny(req, "DESTINATION_NOT_ADMITTED")
        if not req.bound_generation or req.bound_generation not in self.store.generations:  # GUARD:G04a bound generation
            return self._deny(req, "SOURCE_GENERATION_INVALID")
        if any(s.sha256 in p.revoked_digests for s in req.sources):             # GUARD:G02 revoked source
            return self._deny(req, "SOURCE_REVOKED")
        req.authorized = True
        return True

    def _hydrate(self, req: Request) -> bool:
        if self._pre_commit_stop(req):
            return False
        if any(s.sha256 in self.policy.revoked_digests for s in req.sources):   # GUARD:G06 revocation at hydration
            return self._deny(req, "SOURCE_REVOKED_AT_HYDRATION")
        if self.store.active != req.bound_generation:                           # GUARD:G04 active generation
            return self._deny(req, "HYDRATION_GENERATION_CHANGED")
        contents = []
        for s in req.sources:
            body = self.store.read(s)
            self._effect(req, "source_read", chunk_id=s.chunk_id)
            if sha(body) != s.sha256:                                           # GUARD:G05 content digest
                return self._deny(req, "SOURCE_CONTENT_DIGEST_INVALID")
            contents.append(body)
        req.payload = "\n".join(contents)
        req.payload_sha256 = sha(req.payload)
        self._move(req, State.HYDRATED)
        return True

    def _commit(self, req: Request) -> bool:
        with self.commit_lock:                                                  # LOCK:L1 commit side
            if self._pre_commit_stop(req):
                return False
            if self.policy.epoch != req.bound_epoch:                            # GUARD:G07 policy epoch
                return self._deny(req, "POLICY_EPOCH_CHANGED")
            if self.store.active != req.bound_generation:                       # GUARD:G10 generation at commit
                return self._deny(req, "GENERATION_CHANGED_BEFORE_COMMIT")
            if req.destination not in self.policy.admitted_destinations:        # GUARD:G11 destination at commit
                return self._deny(req, "DESTINATION_CHANGED")
            if self.commit_hook is not None:
                self.commit_hook(req)          # test seam INSIDE the critical section
            rec = self._effect(req, "egress_release", destination=req.destination.endpoint,
                               payload_sha256=req.payload_sha256, epoch=req.bound_epoch,
                               generation=req.bound_generation)
            req.commit_seq = rec["seq"]
            self._move(req, State.COMMITTED)
        return True

    # -- post-commit steps -------------------------------------------------- #
    def _dispatch(self, req: Request) -> bool:
        # Deliberately no policy check: COMMIT was the last one.  The send is
        # outside the lock and is never claimed to be atomic with that check.
        self._effect(req, "provider_send", idempotency_key=req.idempotency_key)
        try:
            pid = self.provider.send(req.request_id, req.idempotency_key, req.payload)
        except ProviderNotSent:
            req.sent = False
            self._move(req, State.FAILED, "PROVIDER_UNAVAILABLE_NOT_SENT")
            return False
        except ProviderOutcomeUnknown:
            req.sent = None
            self._move(req, State.UNCERTAIN, "DISPATCH_OUTCOME_UNKNOWN")
            return False
        req.sent, req.provider_request_id = True, pid
        self._audit(req, "worker_ack", provider_request_id=pid)
        self._move(req, State.DISPATCHED)
        return True

    def _stop_cause(self, req: Request) -> Optional[str]:
        if req.revoked_after_commit:                                            # GUARD:G12 revoked after commit
            return "REVOKED_AFTER_COMMIT"
        if req.cancel_cause is not None:                                        # GUARD:G13 cancel after commit
            return req.cancel_cause
        if self.clock.now() >= req.deadline:                                    # GUARD:G14 deadline after commit
            return "DEADLINE_EXPIRED"
        return None

    def _request_cancel(self, req: Request, cause: str) -> bool:
        self._effect(req, "cancel_request", cause=cause)
        req.cancel_requested_to_provider = True
        try:
            acked = self.provider.cancel(req.provider_request_id) is True
        except Exception:
            acked = False
        if acked:                                                               # GUARD:G18 confirmed only on ack
            req.cancel_confirmed, req.confirmed_via = True, "PROVIDER_CANCEL_ACK"
            self._move(req, State.CANCELLED, cause)
        else:
            self._move(req, State.UNCERTAIN, cause)
        return False

    def _stream_one(self, req: Request):
        """True = token forwarded, None = provider finished, False = stopped."""
        cause = self._stop_cause(req)
        if cause is not None:
            return self._request_cancel(req, cause)
        try:
            tok = self.provider.next_token(req.provider_request_id)
        except ProviderFailed:
            self._move(req, State.FAILED, "PROVIDER_FAILED_AFTER_ACK")
            return False
        if tok is None:
            return None
        cause = self._stop_cause(req)
        if cause is not None:                                                   # GUARD:G16 re-check after provider wait
            req.dropped_tokens += 1
            return self._request_cancel(req, cause)
        if req.state is State.DISPATCHED:
            self._move(req, State.STREAMING)
        self._effect(req, "stream_out", text=tok)
        req.forwarded.append(tok)
        return True

    def _completion_commit(self, req: Request) -> bool:
        if req.state is State.DISPATCHED:          # provider produced zero tokens
            self._move(req, State.STREAMING)
        with self.commit_lock:
            if req.revoked_after_commit or (req.digests & self.policy.revoked_digests):  # GUARD:G17 completion revocation
                req.cancel_confirmed, req.confirmed_via = True, "PROVIDER_FINISHED"
                self._move(req, State.CANCELLED, "REVOKED_AFTER_COMMIT")
                return False
            text = "".join(req.forwarded)
            key = sha(f"{req.bound_generation}|{req.payload_sha256}|{req.destination.endpoint}")
            self._effect(req, "cache_write", key=key)
            self.cache[key] = {"text": text, "provenance": set(req.digests), "valid": True,
                               "epoch": req.bound_epoch, "generation": req.bound_generation}
            self._effect(req, "history_write")
            self.history.append({"request_id": req.request_id, "text": text,
                                 "provenance": set(req.digests), "valid": True,
                                 "delivery_confirmed": req.client_connected})
            self._move(req, State.COMPLETED)
        return True

    def _receipt(self, req: Request) -> dict:
        return self._audit(req, "receipt", state=req.state.value, reason=req.reason,
                           payload_sha256=req.payload_sha256, epoch=req.bound_epoch,
                           generation=req.bound_generation, sent=req.sent,
                           forwarded_tokens=len(req.forwarded))

    # -- late callbacks, derived artifacts, cache reads, retry -------------- #
    def on_provider_callback(self, req: Request, token: str) -> bool:
        """A push callback from the provider (late chunk, webhook)."""
        if req.state in TERMINAL or req.state is State.UNCERTAIN:              # GUARD:G19 late callback
            req.late_callbacks_dropped += 1
            self._audit(req, "late_callback_dropped", state=req.state.value)
            return False
        if self._stop_cause(req) is not None:
            req.dropped_tokens += 1
            return False
        if req.state is State.DISPATCHED:
            self._move(req, State.STREAMING)
        self._effect(req, "stream_out", text=token)
        req.forwarded.append(token)
        return True

    def derive(self, name: str, parents: list) -> Optional[dict]:
        prov: set = set()
        for p in parents:
            entry = self.cache.get(p) or self.derived.get(p)
            if entry is None or not entry["valid"]:                             # GUARD:G22 invalid parent
                self._audit(None, "derive_refused", name=name, parent=str(p))
                return None
            prov |= entry["provenance"]
        if prov & self.policy.revoked_digests:                                  # GUARD:G23 revoked provenance
            self._audit(None, "derive_refused", name=name, parent="REVOKED_PROVENANCE")
            return None
        self.derived[name] = {"provenance": prov, "valid": True, "parents": list(parents)}
        self._append("effect:derived_write", None, name=name)
        return self.derived[name]

    def cache_get(self, key: str) -> Optional[str]:
        entry = self.cache.get(key)
        if entry is None or not entry["valid"]:                                 # GUARD:G24 cache read validity
            return None
        return entry["text"]

    def observe_worker_done(self, req: Request) -> bool:
        """Resolve UNCERTAIN(cancel requested) once worker termination is observed."""
        if req.state is not State.UNCERTAIN or not req.cancel_requested_to_provider:
            return False
        if req.provider_request_id in self.provider.finished:                   # GUARD:G25 observed termination
            req.cancel_confirmed, req.confirmed_via = True, "WORKER_TERMINATED"
            self._move(req, State.CANCELLED, req.reason)
            return True
        return False

    def retry(self, req: Request) -> Request:
        """UNCERTAIN dispatch: only a verified-idempotent re-send with the SAME key.
        FAILED and known not sent: a brand-new request through full admission."""
        if req.state is State.UNCERTAIN and req.reason == "DISPATCH_OUTCOME_UNKNOWN":
            if not self.provider.idempotency_verified:                          # GUARD:G26 idempotency verified
                self._audit(req, "retry_refused", reason="IDEMPOTENCY_UNVERIFIED")
                raise RetryRefused("uncertain dispatch; provider idempotency not verified")
            with self.commit_lock:
                if self.policy.epoch != req.bound_epoch:                        # GUARD:G27 policy unchanged for resend
                    self._audit(req, "retry_refused", reason="POLICY_EPOCH_CHANGED")
                    raise RetryRefused("policy changed since commit")
            self._effect(req, "provider_send", idempotency_key=req.idempotency_key, resend=True)
            try:
                pid = self.provider.send(req.request_id, req.idempotency_key, req.payload)
            except (ProviderNotSent, ProviderOutcomeUnknown) as exc:
                raise RetryRefused("provider still unavailable; request remains UNCERTAIN") from exc
            req.sent, req.provider_request_id = True, pid
            self._audit(req, "worker_ack", provider_request_id=pid, resend=True)
            self._move(req, State.DISPATCHED)
            return req
        if req.state is State.FAILED and req.sent is False:
            return self.admit(req.principal, req.sources, authenticated=req.authenticated,
                              destination=req.destination, deadline=self.clock.now() + 100.0)
        raise RetryRefused(f"no retry from {req.state.value}/{req.reason}")


# --------------------------------------------------------------------------- #
# Deterministic interleaving scheduler
# --------------------------------------------------------------------------- #
class Scheduler:
    """Drive lifecycle generators and fire injected callables at named points.

    ``injections`` maps "<point>" (every request) or "<label>:<point>" to a
    list of zero-argument callables run just BEFORE that step executes.
    ``schedule`` is an explicit list of labels; each entry advances that
    request by one step.  Requests still live afterwards are drained in label
    order.  Same inputs -> same trace.
    """

    def __init__(self, injections: Optional[dict] = None):
        self.injections = injections or {}
        self.trace: list = []

    def _fire(self, label: str, point: str) -> None:
        self.trace.append((label, point))
        for key in (point, f"{label}:{point}"):
            for fn in self.injections.get(key, []):
                fn()

    def run(self, gens: dict, schedule: Optional[list] = None) -> list:
        live = dict(gens)

        def step(label):
            try:
                point = next(live[label])
            except StopIteration:
                live.pop(label, None)
                return
            self._fire(label, point)

        for label in schedule or []:
            if label in live:
                step(label)
        for label in sorted(gens):
            while label in live:
                step(label)
        return self.trace


def world(*, tokens=("alpha ", "beta ", "gamma"), mode="ok", supports_cancel=True,
          idempotency_verified=False) -> Controller:
    """Synthetic controller: generations g0 (inactive, older) and g1 (active)."""
    store = GenerationStore()
    store.publish("g0", {"c1": "synthetic source one", "c2": "synthetic source two (old)"}, activate=False)
    store.publish("g1", {"c1": "synthetic source one", "c2": "synthetic source two"})
    dest = Destination("local-fixture", "fixture-model-1", "LOCAL", "inproc://fixture")
    policy = Policy(admitted_destinations={dest})
    provider = FakeProvider(tokens, mode=mode, supports_cancel=supports_cancel,
                            idempotency_verified=idempotency_verified)
    return Controller(store, policy, provider, dest, Clock())
