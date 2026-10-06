"""Behavioral gates, the fail-closed probe suite and the metamorphic grid (PREREGISTRATION s.10).

Every gate is a value in [0, 1]; the known-good reference is v2.
  g_integrity       1 if the kernel accepts the model + receipt pair, else 0
  g_failclosed      fraction of FAILCLOSED_PROBES refused (the callable raised)
  g_truth_sign      sign agreement with generator truth, k/8
  g_monotone        fraction of the pre-declared metamorphic monotonicity checks passed
  g_validation      min(1, BA_val / 0.70)
  g_calibration     1 if ECE_val <= 0.05, else max(0, 1 - (ECE_val - 0.05) / 0.10)
  g_baseline_valid  1 if the section 10 preconditions hold, else 0

Metamorphic grid (fixed; declared here before any v2 model exists)
  base points: the full product of
      the five health flags (listener_running, tls_enabled, peer_allowlist_configured,
      ledger_integrity_ok, configuration_valid) in {0, 1}              -> 32 combinations
      queue_utilization          in (0.0, 0.25, 0.5, 0.75, 1.0)
      consecutive_failures       in (0, 1, 2, 5, 20)
      seconds_since_last_success in (0.0, 60.0, 900.0, 3600.0, 86400.0)
  = 4,000 points.  Checks, for every base point:
      each continuous feature not at its grid maximum is raised to its next grid value;
      each health flag equal to 1 is switched to 0;
  and each check passes iff score(changed) >= score(base) (non-strict).  Every changed point is
  itself a grid point, so the 4,000 scores are computed once.  Total: 19,600 checks.
  The score is the operator-attention score the advisory reports (calibrated).
Fail-closed probes: CONTRACT section 2.6 classes (missing field, extra field, NaN, +/-inf,
out-of-range, text instead of number, and prohibited field names) plus v2 variants (case,
hyphen, space, token and substring variants; a prohibited name replacing a field; nested and
wrapper-level prohibited names; unwrapped or malformed payloads).  Field names are used with
dummy numeric values only.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Callable, Mapping, Sequence

from . import truth

HEALTH_FLAGS = (
    "listener_running",
    "tls_enabled",
    "peer_allowlist_configured",
    "ledger_integrity_ok",
    "configuration_valid",
)
MONOTONE_GRID = {
    "queue_utilization": (0.0, 0.25, 0.5, 0.75, 1.0),
    "consecutive_failures": (0, 1, 2, 5, 20),
    "seconds_since_last_success": (0.0, 60.0, 900.0, 3600.0, 86400.0),
}
GATE_NAMES = (
    "g_integrity",
    "g_failclosed",
    "g_truth_sign",
    "g_monotone",
    "g_validation",
    "g_calibration",
    "g_baseline_valid",
)
VALIDATION_BA_REFERENCE = 0.70
ECE_TOLERANCE = 0.05
ECE_SLOPE_WIDTH = 0.10

# A valid wrapped example (the v1 example input values).
VALID_FEATURES = {
    "listener_running": 1,
    "tls_enabled": 1,
    "peer_allowlist_configured": 1,
    "queue_utilization": 0.12,
    "consecutive_failures": 0,
    "seconds_since_last_success": 14.0,
    "ledger_integrity_ok": 1,
    "configuration_valid": 1,
}
V1_PROHIBITED_NAMES = (
    "address", "date_of_birth", "dob", "email", "fhir", "hl7", "mrn", "name", "obx", "order",
    "order_id", "patient", "patient_id", "phone", "pid", "result", "result_value", "specimen",
    "specimen_id", "ssn",
)
PROHIBITED = "prohibited non-operational field"


def _with(**changes) -> dict:
    features = dict(VALID_FEATURES)
    for key, value in changes.items():
        features[key] = value
    return {"features": features}


def _extra(name: str, value: Any = 0) -> dict:
    payload = _with()
    payload["features"][name] = value
    return payload


def _without(name: str) -> dict:
    payload = _with()
    del payload["features"][name]
    return payload


def _deep(levels: int) -> dict:
    node: dict = {"depth_marker": 0}
    for _ in range(levels):
        node = {"level": node}
    payload = _with()
    payload["features"]["nested"] = node
    return payload


def _replace(field: str, name: str) -> dict:
    payload = _without(field)
    payload["features"][name] = VALID_FEATURES[field]
    return payload


def build_probes() -> list[tuple[str, Any, str]]:
    """(probe_id, payload, expected message fragment); every probe must be refused."""
    nan, inf = float("nan"), float("inf")
    probes: list[tuple[str, Any, str]] = [
        ("missing_field", _without("tls_enabled"), "missing=['tls_enabled']"),
        ("extra_field", _extra("uptime_seconds", 5), "unexpected=['uptime_seconds']"),
        ("nan_continuous", _with(queue_utilization=nan), "queue_utilization must be finite"),
        ("inf_continuous", _with(seconds_since_last_success=inf), "seconds_since_last_success must be finite"),
        ("neg_inf_continuous", _with(consecutive_failures=-inf), "consecutive_failures must be finite"),
        ("nan_binary", _with(tls_enabled=nan), "tls_enabled must be boolean or integer 0/1"),
        ("out_of_range_high", _with(queue_utilization=1.5), "queue_utilization must be between 0 and 1"),
        ("out_of_range_low", _with(seconds_since_last_success=-1.0), "must be between 0 and 86400"),
        ("out_of_range_failures", _with(consecutive_failures=21), "must be between 0 and 20"),
        ("huge_integer", _with(consecutive_failures=10**400), "consecutive_failures must be a finite number"),
        ("text_instead_of_number", _with(queue_utilization="0.5"), "text values are not accepted"),
        ("text_instead_of_binary", _with(tls_enabled="1"), "text values are not accepted"),
        ("bool_for_continuous", _with(queue_utilization=True), "queue_utilization must be a finite number"),
        ("binary_not_0_1", _with(listener_running=2), "listener_running must be boolean or integer 0/1"),
        ("binary_float", _with(listener_running=1.0), "listener_running must be boolean or integer 0/1"),
        ("null_value", _with(configuration_valid=None), "configuration_valid must be boolean or integer 0/1"),
        ("list_value", _with(queue_utilization=[0.5]), "queue_utilization must be a finite number"),
        ("unwrapped_features", dict(VALID_FEATURES), 'input must be wrapped as {"features": {...}}'),
        ("extra_wrapper_key", {**_with(), "note_count": 0}, "may contain only the features field"),
        ("features_not_object", {"features": [0, 1]}, "features must be a JSON object"),
        ("top_level_not_object", [_with()], "input must be a JSON object"),
        ("too_deep", _deep(40), "input nesting exceeds"),
        ("unscreened_extra_name", _extra("sortorder", 0), "unexpected=['sortorder']"),
    ]
    # CONTRACT 2.6: field names containing patient, hl7, fhir, specimen, order, result, mrn,
    # plus every v1 exact name.
    for name in V1_PROHIBITED_NAMES:
        probes.append((f"exact_{name}", _extra(name), PROHIBITED))
    variants = {
        "case_upper_patient": "PATIENT",
        "case_mixed_hl7": "Hl7",
        "hyphen_patient_id": "Patient-ID",
        "space_order_id": "order id",
        "padded_mrn": "  MRN  ",
        "hyphen_result_value": "Result-Value",
        "token_hl7": "source_hl7_port",
        "token_order": "last_order_count",
        "token_result": "result_code",
        "token_name": "queue_name",
        "token_email": "operator-email",
        "token_pid": "pid file",
        "token_dob": "dob_year",
        "substring_patient": "PatientCount",
        "substring_hl7": "xhl7x",
        "substring_fhir": "fhirendpoint",
        "substring_specimen": "specimens",
        "substring_mrn": "mrnhash",
    }
    for probe_id, name in variants.items():
        probes.append((probe_id, _extra(name), PROHIBITED))
    probes.extend(
        [
            ("prohibited_replaces_field", _replace("queue_utilization", "patient_id"), PROHIBITED),
            ("prohibited_replaces_binary", _replace("tls_enabled", "Specimen-ID"), PROHIBITED),
            ("nested_prohibited", _extra("meta", {"mrn": 0}), PROHIBITED),
            ("nested_list_prohibited", _extra("tags", [{"specimen_id": 0}]), PROHIBITED),
            ("wrapper_level_prohibited", {**_with(), "patient": 0}, PROHIBITED),
        ]
    )
    ids = [p[0] for p in probes]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate probe id")
    return probes


FAILCLOSED_PROBES = build_probes()


def run_failclosed_probes(advise: Callable[[Any], Any]) -> dict:
    """A probe is refused iff ``advise`` raises.  Returns counts, escapes and messages."""
    refused, escaped, messages = 0, [], {}
    for probe_id, payload, _fragment in FAILCLOSED_PROBES:
        try:
            advise(copy.deepcopy(payload))
        except Exception as exc:  # noqa: BLE001 -- any refusal counts; the type is recorded
            refused += 1
            messages[probe_id] = f"{type(exc).__name__}: {exc}"
        else:
            escaped.append(probe_id)
    total = len(FAILCLOSED_PROBES)
    return {"refused": refused, "total": total, "fraction": refused / total, "escaped": escaped,
            "messages": messages}


def monotone_points() -> list[dict]:
    points = []
    for flags in itertools.product((0, 1), repeat=len(HEALTH_FLAGS)):
        for q, f, s in itertools.product(*MONOTONE_GRID.values()):
            point = dict(zip(HEALTH_FLAGS, flags))
            point.update(
                {"queue_utilization": q, "consecutive_failures": f, "seconds_since_last_success": s}
            )
            points.append(point)
    return points


def _key(point: Mapping) -> tuple:
    return tuple(point[name] for name in sorted(point))


def monotone_checks() -> list[tuple[str, dict, dict]]:
    checks = []
    for point in monotone_points():
        for name, grid in MONOTONE_GRID.items():
            index = grid.index(point[name])
            if index + 1 < len(grid):
                changed = dict(point)
                changed[name] = grid[index + 1]
                checks.append((f"raise:{name}", point, changed))
        for flag in HEALTH_FLAGS:
            if point[flag] == 1:
                changed = dict(point)
                changed[flag] = 0
                checks.append((f"flag_off:{flag}", point, changed))
    return checks


def g_monotone(score: Callable[[Mapping], float]) -> tuple[float, dict]:
    """``score(features) -> float``.  Returns (fraction passed, report)."""
    cache = {_key(p): score(dict(p)) for p in monotone_points()}
    failed_by_kind: dict[str, int] = {}
    checks = monotone_checks()
    passed = 0
    for kind, base, changed in checks:
        if cache[_key(changed)] >= cache[_key(base)]:
            passed += 1
        else:
            failed_by_kind[kind] = failed_by_kind.get(kind, 0) + 1
    return passed / len(checks), {"checks": len(checks), "passed": passed,
                                  "failed_by_kind": dict(sorted(failed_by_kind.items()))}


def g_failclosed(advise: Callable[[Any], Any]) -> tuple[float, dict]:
    report = run_failclosed_probes(advise)
    return report["fraction"], report


def g_truth_sign(weights: Mapping[str, float] | Sequence[float]) -> float:
    return truth.sign_gate(weights)


def g_validation(balanced_accuracy: float | None) -> float:
    if balanced_accuracy is None:
        return 0.0
    return min(1.0, balanced_accuracy / VALIDATION_BA_REFERENCE)


def g_calibration(ece: float | None) -> float:
    if ece is None:
        return 0.0
    if ece <= ECE_TOLERANCE:
        return 1.0
    return max(0.0, 1.0 - (ece - ECE_TOLERANCE) / ECE_SLOPE_WIDTH)


def g_baseline_valid(validity: Mapping) -> float:
    return 1.0 if validity.get("valid") is True else 0.0


def g_integrity(model_path, receipt_path, kernel_module) -> tuple[float, str | None]:
    try:
        kernel_module.OpsHealthKernel(model_path, receipt_path)
    except kernel_module.OperationalModelError as exc:
        return 0.0, str(exc)
    return 1.0, None


def gate_vector(values: Mapping[str, float]) -> dict[str, float]:
    missing = [name for name in GATE_NAMES if name not in values]
    if missing:
        raise ValueError(f"missing gates: {missing}")
    out = {}
    for name in GATE_NAMES:
        value = float(values[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} outside [0, 1]")
        out[name] = value
    return out
