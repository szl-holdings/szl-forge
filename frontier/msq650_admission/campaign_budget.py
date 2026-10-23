"""Pure, fail-closed MSQ650 reservation accounting; never runtime authority.

observe_reservation accepts three UTF-8 JSON byte strings with these exact shapes:
 ledger = {"schema":"szl.msq650.campaign-ledger.v1", "genesis": {
   "campaign_id":str, "candidate_sha256":hex64, "start_epoch":int,
   "deadline_epoch":int, "max_cost_cents":int, "max_wall_seconds":int},
   "records":[record, ...]}
 request = {"lease_id":str, "candidate_sha256":hex64, "start_epoch":int,
   "wall_seconds":int, "rate_cents_per_hour":int, "overhead_cents":int}
 context = {"expected_campaign_id":str, "expected_candidate_sha256":hex64,
   "expected_start_epoch":int, "expected_genesis_sha256":hex64,
   "expected_tip_sha256":hex64, "expected_sequence":int,
   "current_epoch":int, "observed_at_epoch":int}
 record = request fields plus sequence, previous_sha256, reserved_cost_cents,
   cumulative_cost_cents, cumulative_wall_seconds, status, sha256.
 Record hashes exclude their own sha256 field. Genesis and record hashes use
 sorted-key compact ASCII JSON. Sequence starts at 1; its predecessor is the
 genesis hash. Context must come from the external campaign custodian, not from
 reading this ledger: its pinned genesis/tip/sequence prevents accepted resets.

Each reservation charges ceil(rate * wall_seconds / 3600) + overhead. Completed,
failed and unknown runs are charged identically; reconciliation/refunds do not
exist here. Reservations cannot overlap, cross the elapsed campaign deadline,
or exceed cumulative caps. Proposed records are NOT persisted or signed.

Caller-supplied expectations, clocks and hashes are comparison inputs, NOT proof
of authority, authentic prices, measured usage, trusted time or exclusive leases.
A separate authenticated atomic custodian and qualified runtime are required.
Two concurrent calls can both be feasible; this function grants neither call.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType

MAX_COST_CENTS = 200_000
MAX_WALL_SECONDS = 180_000
MAX_RECORDS = 256
MAX_LEDGER_BYTES = 131_072
MAX_INPUT_BYTES = 16_384
MAX_CONTEXT_AGE_SECONDS = 60
MAX_INTEGER = (1 << 63) - 1
GENESIS_FIELDS = {"campaign_id", "candidate_sha256", "start_epoch", "deadline_epoch",
                  "max_cost_cents", "max_wall_seconds"}
REQUEST_FIELDS = {"lease_id", "candidate_sha256", "start_epoch", "wall_seconds",
                  "rate_cents_per_hour", "overhead_cents"}
RECORD_FIELDS = REQUEST_FIELDS | {"sequence", "previous_sha256", "reserved_cost_cents",
    "cumulative_cost_cents", "cumulative_wall_seconds", "status", "sha256"}
CONTEXT_FIELDS = {"expected_campaign_id", "expected_candidate_sha256",
    "expected_start_epoch", "expected_genesis_sha256", "expected_tip_sha256",
    "expected_sequence", "current_epoch", "observed_at_epoch"}


class BudgetError(ValueError):
    """Invalid, stale, unbound or infeasible modeled reservation."""


@dataclass(frozen=True)
class CampaignObservation:
    """Immutable comparison result, deliberately unusable as a truthy gate."""

    deadline_epoch: int
    proposed_record: object
    schema: str = field(default="szl.msq650.campaign-observation.v1", init=False)
    evidence_class: str = field(default="MODELED", init=False)
    feasible: bool = field(default=True, init=False)
    execution_authority: bool = field(default=False, init=False)
    training_admitted: bool = field(default=False, init=False)
    atomic_reservation: bool = field(default=False, init=False)
    externally_authenticated: bool = field(default=False, init=False)

    def __post_init__(self):
        object.__setattr__(self, "proposed_record", MappingProxyType(dict(self.proposed_record)))

    def __bool__(self):
        raise TypeError("MODELED_CAMPAIGN_OBSERVATION_IS_NOT_ADMISSION")


def canonical_sha256(value):
    """Content digest only; this function is not a signer or trust anchor."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _fail(message):
    raise BudgetError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _forbidden_number(_value):
    _fail("NON_INTEGER_JSON_NUMBER")


def _load(raw, maximum):
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        _fail("INVALID_INPUT_BYTES")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                           parse_float=_forbidden_number, parse_constant=_forbidden_number)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise BudgetError("INVALID_JSON") from exc
    if type(value) is not dict:
        _fail("EXPECTED_OBJECT")
    return value


def _fields(value, expected):
    if type(value) is not dict or set(value) != expected:
        _fail("INVALID_FIELDS")


def _integer(value, minimum=0, maximum=MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail("INVALID_INTEGER")
    return value


def _identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}", value) is None:
        _fail("INVALID_IDENTIFIER")


def _hash(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail("INVALID_HASH")


def _request(value):
    _fields(value, REQUEST_FIELDS)
    _identifier(value["lease_id"])
    _hash(value["candidate_sha256"])
    _integer(value["start_epoch"], 1)
    _integer(value["wall_seconds"], 1, MAX_WALL_SECONDS)
    _integer(value["rate_cents_per_hour"], 1, MAX_COST_CENTS * 3600)
    _integer(value["overhead_cents"], 0, MAX_COST_CENTS)


def _cost(value):
    return (value["rate_cents_per_hour"] * value["wall_seconds"] + 3599) // 3600 + value["overhead_cents"]


def observe_reservation(ledger_bytes, request_bytes, context_bytes):
    """Return MODELED feasibility and an unsigned proposal, or raise BudgetError.

    No files, processes, networking, clocks, signing keys or runtime APIs used.
    A valid result always has training_admitted = execution_authority = False.
    """
    ledger = _load(ledger_bytes, MAX_LEDGER_BYTES)
    request = _load(request_bytes, MAX_INPUT_BYTES)
    context = _load(context_bytes, MAX_INPUT_BYTES)
    _fields(ledger, {"schema", "genesis", "records"})
    _fields(context, CONTEXT_FIELDS)
    if ledger["schema"] != "szl.msq650.campaign-ledger.v1":
        _fail("INVALID_SCHEMA")
    _request(request)
    _identifier(context["expected_campaign_id"])
    for key in ("expected_candidate_sha256", "expected_genesis_sha256", "expected_tip_sha256"):
        _hash(context[key])
    _integer(context["expected_sequence"], 0, MAX_RECORDS)
    for key in ("expected_start_epoch", "current_epoch", "observed_at_epoch"):
        _integer(context[key], 1)
    now = context["current_epoch"]
    if not 0 <= now - context["observed_at_epoch"] <= MAX_CONTEXT_AGE_SECONDS:
        _fail("STALE_CONTEXT")
    genesis = ledger["genesis"]
    _fields(genesis, GENESIS_FIELDS)
    _identifier(genesis["campaign_id"])
    _hash(genesis["candidate_sha256"])
    _integer(genesis["start_epoch"], 1)
    _integer(genesis["deadline_epoch"], 1)
    _integer(genesis["max_cost_cents"], 1, MAX_COST_CENTS)
    _integer(genesis["max_wall_seconds"], 1, MAX_WALL_SECONDS)
    if genesis["deadline_epoch"] != genesis["start_epoch"] + genesis["max_wall_seconds"]:
        _fail("INVALID_DEADLINE")
    if not genesis["start_epoch"] <= now < genesis["deadline_epoch"]:
        _fail("CAMPAIGN_NOT_CURRENT")
    for key in ("campaign_id", "candidate_sha256", "start_epoch"):
        if genesis[key] != context["expected_" + key]:
            _fail("CAMPAIGN_BINDING_MISMATCH")
    tip = canonical_sha256(genesis)
    if tip != context["expected_genesis_sha256"]:
        _fail("GENESIS_COMMITMENT_MISMATCH")
    records = ledger["records"]
    if type(records) is not list or len(records) > MAX_RECORDS:
        _fail("INVALID_RECORD_COUNT")
    if len(records) != context["expected_sequence"]:
        _fail("SEQUENCE_ROLLBACK_OR_MISMATCH")
    total_cost = total_wall = 0
    previous_end = genesis["start_epoch"]
    leases = set()

    def check_reservation(item):
        if item["candidate_sha256"] != genesis["candidate_sha256"]:
            _fail("CANDIDATE_DRIFT")
        if item["lease_id"] in leases:
            _fail("DUPLICATE_LEASE")
        if item["start_epoch"] < previous_end:
            _fail("OVERLAPPING_RESERVATION")
        if item["start_epoch"] + item["wall_seconds"] > genesis["deadline_epoch"]:
            _fail("CAMPAIGN_DEADLINE_EXCEEDED")
        if total_cost + _cost(item) > genesis["max_cost_cents"]:
            _fail("COST_CAP_EXCEEDED")
        if total_wall + item["wall_seconds"] > genesis["max_wall_seconds"]:
            _fail("WALL_CAP_EXCEEDED")

    for sequence, item in enumerate(records, 1):
        _fields(item, RECORD_FIELDS)
        _request({key: item[key] for key in REQUEST_FIELDS})
        for key in ("sequence", "reserved_cost_cents", "cumulative_cost_cents", "cumulative_wall_seconds"):
            _integer(item[key], 1)
        for key in ("previous_sha256", "sha256"):
            _hash(item[key])
        if type(item["status"]) is not str or item["status"] not in {"reserved", "failed", "unknown", "completed"}:
            _fail("INVALID_STATUS")
        if item["sequence"] != sequence or item["previous_sha256"] != tip:
            _fail("BROKEN_CHAIN")
        if canonical_sha256({key: value for key, value in item.items() if key != "sha256"}) != item["sha256"]:
            _fail("RECORD_HASH_MISMATCH")
        check_reservation(item)
        total_cost += _cost(item)
        total_wall += item["wall_seconds"]
        if (item["reserved_cost_cents"], item["cumulative_cost_cents"], item["cumulative_wall_seconds"]) != (_cost(item), total_cost, total_wall):
            _fail("ACCOUNTING_MISMATCH")
        leases.add(item["lease_id"])
        previous_end = item["start_epoch"] + item["wall_seconds"]
        tip = item["sha256"]
    if tip != context["expected_tip_sha256"]:
        _fail("TIP_ROLLBACK_OR_MISMATCH")
    if len(records) >= MAX_RECORDS:
        _fail("RECORD_LIMIT_REACHED")
    if request["start_epoch"] < now:
        _fail("REQUEST_IN_PAST")
    check_reservation(request)
    proposed = dict(request, sequence=len(records) + 1, previous_sha256=tip,
                    reserved_cost_cents=_cost(request), cumulative_cost_cents=total_cost + _cost(request),
                    cumulative_wall_seconds=total_wall + request["wall_seconds"], status="reserved")
    proposed["sha256"] = canonical_sha256(proposed)
    return CampaignObservation(genesis["deadline_epoch"], proposed)
