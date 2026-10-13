"""Independent, provider-free byte-accounting oracle; not a model launcher.

MODELED reservations only. A fitting plan never grants training or publication.
Prior art: https://github.com/lyogavin/airllm and
https://www.deepspeed.ai/2022/09/09/zero-inference.html.
No upstream implementation is copied. No tensor files are opened or verified.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

PHASES = ("construction", "forward", "backward", "optimizer", "checkpoint")
TIERS = ("gpu", "host", "disk")


class InputError(ValueError):
    """Malformed declarations cannot be used as a capacity plan."""


def _bytes(value, field, unknown=True):
    if value is None and unknown:
        return
    if type(value) is not int or value < 0:
        raise InputError(f"{field}: expected nonnegative integer bytes")


def _name(value, field):
    if type(value) is not str or not value.strip():
        raise InputError(f"{field}: expected nonempty text")


def _targets(values, field):
    if type(values) is not tuple:
        raise InputError(f"{field}: expected immutable tuple")
    for value in values:
        _name(value, field)
    if len(values) != len(set(values)):
        raise InputError(f"{field}: duplicate target")


@dataclass(frozen=True)
class Budget:
    capacity_bytes: int | None
    reserve_bytes: int | None = None

    def __post_init__(self):
        _bytes(self.capacity_bytes, "capacity")
        _bytes(self.reserve_bytes, "reserve")


@dataclass(frozen=True)
class Reservation:
    name: str
    owner: str
    tier: str
    size_bytes: int | None
    phases: tuple[str, ...]
    persistent: bool = False
    already_allocated: bool = False
    device: str | None = None
    role: str = "workspace"

    def __post_init__(self):
        _name(self.name, "reservation name")
        _name(self.owner, "owner")
        _name(self.role, "role")
        if self.tier not in TIERS:
            raise InputError("unknown memory tier")
        _bytes(self.size_bytes, "reservation size")
        if type(self.phases) is not tuple or not self.phases:
            raise InputError("reservation requires explicit immutable phase lifetime")
        if len(self.phases) != len(set(self.phases)) or any(p not in PHASES for p in self.phases):
            raise InputError("invalid or repeated phase")
        if type(self.persistent) is not bool or type(self.already_allocated) is not bool:
            raise InputError("lifetime/allocation flags must be exact booleans")
        if self.already_allocated and (self.tier != "disk" or not self.persistent):
            raise InputError("only persistent disk assets may be already allocated")
        if self.persistent:
            first_phase = min(PHASES.index(p) for p in self.phases)
            if set(self.phases) != set(PHASES[first_phase:]):
                raise InputError("persistent allocation must remain through every later phase")
        if self.already_allocated and set(self.phases) != set(PHASES):
            raise InputError("existing disk assets must be accounted in every phase")
        if self.tier == "gpu":
            if type(self.device) is not str or re.fullmatch(r"cuda:[0-9]+", self.device) is None:
                raise InputError("GPU reservation requires an explicit cuda:N device")
        elif self.device is not None:
            raise InputError("non-GPU reservation cannot select a CUDA device")
        if self.role == "original":
            if self.tier != "disk" or not self.persistent or set(self.phases) != set(PHASES):
                raise InputError("originals must remain persistent on disk in every phase")


def _sum_known(values):
    return None if any(v is None for v in values) else sum(values)


def evaluate_plan(reservations, budgets, *, base_id, base_revision,
                  target_device, expected_adapter_targets=(), actual_adapter_targets=()):
    """Account supplied declarations, including construction and checkpoint peaks.

    Disk capacity is *currently free* space. Existing retained disk assets remain
    in gross footprint, but are not charged to free space a second time.
    GPU/host capacity is the caller's available reservation budget, not total RAM.
    Phases are serial, not overlapping sessions; prefetch overlap within a phase
    must be supplied as separate simultaneous reservations. No scheduler is run.
    Target coverage is caller-declared, not inspected from actual tensor files.
    Manifest completeness and all runtime/backend qualification remain UNKNOWN.
    """
    _name(base_id, "base id")
    if type(base_revision) is not str or re.fullmatch(r"[0-9a-f]{40}", base_revision) is None:
        raise InputError("base revision must be an immutable lowercase SHA-1 identity")
    if type(target_device) is not str or re.fullmatch(r"cuda:[0-9]+", target_device) is None:
        raise InputError("target device must be explicit")
    if type(reservations) is not tuple or not reservations:
        raise InputError("an explicit immutable nonempty reservation manifest is required")
    if any(type(r) is not Reservation for r in reservations):
        raise InputError("invalid reservation object")
    if len({r.name for r in reservations}) != len(reservations):
        raise InputError("duplicate reservation name")
    if type(budgets) is not dict or set(budgets) != set(TIERS):
        raise InputError("all three tier budgets must be explicit")
    if any(type(budgets[t]) is not Budget for t in TIERS):
        raise InputError("invalid budget object")
    _targets(expected_adapter_targets, "expected adapter targets")
    _targets(actual_adapter_targets, "actual adapter targets")
    if set(expected_adapter_targets) != set(actual_adapter_targets):
        raise InputError("declared adapter coverage differs from actual target set")
    if any(r.tier == "gpu" and r.device != target_device for r in reservations):
        raise InputError("reservation escaped the explicit target device")
    request = {
        "base_id": base_id, "base_revision": base_revision,
        "target_device": target_device,
        "reservations": [asdict(r) for r in reservations],
        "budgets": {t: asdict(budgets[t]) for t in TIERS},
        "expected_adapter_targets": sorted(expected_adapter_targets),
        "actual_adapter_targets": sorted(actual_adapter_targets),
    }
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
    rows, excesses, unknowns = [], [], set()
    for phase in PHASES:
        live = [r for r in reservations if phase in r.phases]
        row = {"phase": phase, "live_reservations": [r.name for r in live],
               "gross_bytes": {}, "required_new_or_reserved_bytes": {}}
        for tier in TIERS:
            allocations = [r for r in live if r.tier == tier]
            row["gross_bytes"][tier] = _sum_known([r.size_bytes for r in allocations])
            charged = [r.size_bytes for r in allocations if not r.already_allocated]
            amounts = charged + [budgets[tier].reserve_bytes]
            required = _sum_known(amounts)
            known_lower_bound = sum(v for v in amounts if v is not None)
            row["required_new_or_reserved_bytes"][tier] = required
            capacity = budgets[tier].capacity_bytes
            # Unknown existing originals also leave manifest accounting incomplete.
            if required is None or capacity is None or row["gross_bytes"][tier] is None:
                unknowns.add(f"{phase}:{tier}:unknown size or budget")
            if capacity is not None and known_lower_bound > capacity:
                excesses.append({"phase": phase, "tier": tier,
                                 "required_bytes": required, "capacity_bytes": capacity,
                                 "known_lower_bound_bytes": known_lower_bound,
                                 "over_bytes_at_least": known_lower_bound - capacity})
        rows.append(row)
    peaks = {}
    for tier in TIERS:
        values = [r["required_new_or_reserved_bytes"][tier] for r in rows]
        peaks[tier] = None if any(v is None for v in values) else max(values)
    decision = "BLOCKED" if excesses else "UNKNOWN" if unknowns else "MODELED"
    return {
        "evidence_class": "MODELED", "decision": decision,
        "fits_declared_budget": False if excesses else None if unknowns else True,
        "request_sha256": request_hash, "request": request,
        "phase_accounting": rows, "peak_required_bytes": peaks,
        "budget_excesses": excesses, "unknowns": sorted(unknowns),
        "manifest_completeness": "UNKNOWN",
        "weight_byte_integrity": "UNKNOWN", "CUDA_qualification": "NOT_RUN",
        "phase_model": "SERIAL_PHASES_WITH_DECLARED_WITHIN_PHASE_OVERLAP_ONLY",
        "exclusive_runtime_reservation": "NOT_RUN",
        "device_identity": "ORDINAL_ONLY_UUID_BINDING_UNKNOWN",
        "model_quality": "UNKNOWN", "numerical_parity": "NOT_RUN",
        "training_eligible": False, "publication_eligible": False,
        "execution_authority": False,
    }


def simulate_abort(reservations, *, owner, phase):
    """Return names only; never release real memory or delete a real asset."""
    _name(owner, "abort owner")
    if phase not in PHASES:
        raise InputError("unknown abort phase")
    if type(reservations) is not tuple or any(type(r) is not Reservation for r in reservations):
        raise InputError("invalid abort manifest")
    if len({r.name for r in reservations}) != len(reservations):
        raise InputError("duplicate abort reservation name")
    released = [r.name for r in reservations
                if r.owner == owner and phase in r.phases and not r.persistent]
    retained = [r.name for r in reservations if r.name not in released]
    return {"evidence_class": "SIMULATED", "released_temporary_names": released,
            "retained_names": retained, "real_deletions": 0,
            "training_eligible": False, "publication_eligible": False,
            "execution_authority": False}
