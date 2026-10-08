# SPDX-License-Identifier: Apache-2.0
"""OAC Ops Health v2 inference kernel: fail-closed and advisory-only (SYNTHETIC).

The kernel scores exactly eight operational transport counters and emits an
operator-attention advisory: ALERT, NO_ALERT or ABSTAIN (Mondrian split-conformal).
It cannot acknowledge messages, release anything, control devices or make care
decisions, and its output authority map is all-false.  Its training data are
synthetic, so no score or rate it produces says anything about a real system.

Attribution
-----------
Input validation (feature specs, prohibited-key screening, finiteness and range checks),
the receipt check and the CLI shape are adapted from the v1 kernel
``oac_operational_health.py`` of SZLHOLDINGS/oac-system-health-v1 at Hub revision
dd7d109813abcd90c5250106dc2eabd2804e7ac3 (Apache License 2.0, Copyright SZL Holdings).
Changes from v1:
  * the v1 exact-name screen is kept and extended with "_"-token and substring screens;
    the screen runs recursively over the whole payload before any schema check;
  * text values anywhere in the payload are refused (the input is numeric only);
  * only the wrapped form {"features": {...}} is accepted;
  * strict JSON: duplicate keys are refused, and non-finite constants are refused in
    artifacts and receipts;
  * a calibrator (none, platt or isotonic) and a Mondrian split-conformal layer
    (ALERT / NO_ALERT / ABSTAIN) are added;
  * the receipt binds the model, this kernel file, the generator, the data manifest, the
    preregistration and the source commit; the kernel hashes its own file.
Stage-1 review hardening (logs/v2/review/STAGE1_REVIEW.md):
  * the prohibited-key screen also matches the Unicode-folded key (NFKC, format characters
    such as zero-width spaces and soft hyphens removed, casefolded), so look-alike keys are
    refused as prohibited and not only by the schema check;
  * refusal messages are bounded: any echoed key (input, model or receipt) shows at most
    MAX_SHOWN_KEY characters, at most MAX_SHOWN_KEYS unexpected keys are listed, an echoed input
    path shows at most MAX_SHOWN_PATH characters, and the CLI prints a refusal as one JSON line
    of at most MAX_SHOWN_ERROR bytes (ASCII, JSON-escaped); values are never echoed (an
    unsupported calibrator kind is refused without showing it);
  * a -0.0 input is normalized to +0.0 and reported contributions carry no negative zero;
  * the registered constants are pinned: conformal alpha 0.10, l2 on the registered grid, and
    the receipt's preregistration digest;
  * malformed provenance of any JSON type is refused cleanly, and the CLI turns any unexpected
    internal error into a refusal (exit 2) without a stack trace.
Stage 2 (registered search): the receipt's selection block also carries the search provenance,
each null or checked for form: search_trace_sha256, receipts_dir_sha256 and amendments_sha256
(64-hex sha256) and conformal_keep (true or false, the section 9 ABL-conformal decision).  The
kernel checks their form only; freeze.verify_origin checks amendments_sha256 against git.

Standard library only.  Licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_SCHEMA = "szl-oac/ops-health-logistic-model/v2"
RECEIPT_SCHEMA = "szl-oac/ops-health-artifact-receipt/v2"
ADVISORY_SCHEMA = "szl-oac/ops-health-advisory/v2"
MODEL_TYPE = "binary_logistic_regression_standard_library"
MODEL_PURPOSE = "synthetic_operational_transport_attention_only"
MODEL_AUTHORITY = "advisory_only_no_ack_release_device_or_care_authority"
SCORE_SEMANTICS = "synthetic_attention_probability_estimate_not_production_calibrated"
RECEIPT_ATTESTS = "integrity and origin only; not quality"
CONFORMAL_METHOD = "mondrian_split_conformal"
TRAINING_ALGORITHM = "newton_irls_l2_logistic_regression"
THRESHOLD_SELECTION = "max_balanced_accuracy_then_f1_then_closest_to_0.5_then_lowest"
THRESHOLD_GRID = "0.01:0.99:0.01"
GENERATOR_VERSION = "ops-health-generator/2.0.0"
V1_HUB_REVISION = "dd7d109813abcd90c5250106dc2eabd2804e7ac3"
V1_BASE_SOURCE = f"SZLHOLDINGS/oac-system-health-v1@{V1_HUB_REVISION}:oac_operational_health.py"
STOP_REASONS = frozenset({"CONVERGED", "CAP_REACHED", "LINE_SEARCH_FAILED"})
CALIBRATOR_KINDS = ("none", "platt", "isotonic")
MAX_ITER = 100
TOL = 1e-10
PARAM_LIMIT = 100.0
PLATT_CLIP = 1e-15
MAX_ISOTONIC_BLOCKS = 8192
MAX_INPUT_DEPTH = 32
MAX_INPUT_BYTES = 1_000_000
MAX_SHOWN_KEY = 64
MAX_SHOWN_KEYS = 8
MAX_SHOWN_PATH = 256
MAX_SHOWN_ERROR = 1024
# Registered constants (v2/PREREGISTRATION.md sections 6 and 12; digest of the registered text).
REGISTERED_ALPHA = 0.10
REGISTERED_L2_GRID = (0.0, 1e-4, 1e-3, 1e-2)
PREREGISTRATION_SHA256 = "145497789c7ad78a80a94cfd7f2b031ca9a20df49c06e0b3412af9070fa1ef4b"

# The output authority map is v1's exact five-key, all-false map.
AUTHORITY_MAP = {
    "acknowledgement": False,
    "clinical_decision": False,
    "device_control": False,
    "result_interpretation": False,
    "result_release": False,
}

# (name, minimum, maximum, kind): identical to the v1 kernel's FEATURE_SPECS.
FEATURE_SPECS = (
    ("listener_running", 0.0, 1.0, "binary"),
    ("tls_enabled", 0.0, 1.0, "binary"),
    ("peer_allowlist_configured", 0.0, 1.0, "binary"),
    ("queue_utilization", 0.0, 1.0, "continuous"),
    ("consecutive_failures", 0.0, 20.0, "continuous"),
    ("seconds_since_last_success", 0.0, 86400.0, "continuous"),
    ("ledger_integrity_ok", 0.0, 1.0, "binary"),
    ("configuration_valid", 0.0, 1.0, "binary"),
)
FEATURE_NAMES = tuple(spec[0] for spec in FEATURE_SPECS)

# v1's exact prohibited names (matched after key normalization).
V1_PROHIBITED_KEYS = frozenset(
    {
        "address", "date_of_birth", "dob", "email", "fhir", "hl7", "mrn", "name", "obx",
        "order", "order_id", "patient", "patient_id", "phone", "pid", "result",
        "result_value", "specimen", "specimen_id", "ssn",
    }
)
# v2 additions: any "_"-separated token of the normalized key, and any substring.
PROHIBITED_TOKENS = frozenset(
    {
        "patient", "hl7", "fhir", "specimen", "order", "result", "mrn", "ssn", "dob",
        "obx", "pid", "email", "phone", "address", "name",
    }
)
PROHIBITED_SUBSTRINGS = ("fhir", "hl7", "mrn", "patient", "specimen")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")

ARTIFACT_KEYS = frozenset(
    {
        "schema", "model_type", "purpose", "authority", "synthetic_training_data",
        "feature_manifest", "intercept", "weights", "calibrator", "decision_threshold",
        "conformal", "score_semantics", "training", "generator",
    }
)
TRAINING_KEYS = frozenset(
    {
        "algorithm", "l2", "iterations", "converged", "stop_reason", "max_iter", "tol",
        "threshold_selection", "threshold_grid", "splits",
    }
)
TRAINING_SPLIT_ROLES = frozenset({"train", "calibration", "conformal", "threshold"})
GENERATOR_KEYS = frozenset({"version", "source_sha256", "master_seed", "row_schema"})
CONFORMAL_KEYS = frozenset({"method", "alpha", "q_hat", "n_calibration"})
RECEIPT_KEYS = frozenset(
    {
        "schema", "purpose", "authority", "model_sha256", "kernel_sha256",
        "generator_sha256", "data_manifest_sha256", "preregistration_sha256", "source",
        "selection", "attests",
    }
)
SOURCE_KEYS = frozenset({"repository", "commit", "tree_clean", "base_v1_source", "v1_hub_revision"})
SELECTION_KEYS = frozenset(
    {
        "procedure", "selection_split", "trials", "selected_trial", "stop_reason",
        "search_trace_sha256", "receipts_dir_sha256", "amendments_sha256", "conformal_keep",
    }
)
# Stage-2 selection provenance digests: each is null or a 64-hex sha256.
SELECTION_DIGEST_KEYS = ("search_trace_sha256", "receipts_dir_sha256", "amendments_sha256")


class OperationalModelError(ValueError):
    """Base error for fail-closed artifact and input validation."""


class ModelArtifactError(OperationalModelError):
    """Raised when a model, receipt or this kernel fails integrity or schema validation."""


class ModelInputError(OperationalModelError):
    """Raised when inference input is outside the operational-only schema."""


# --------------------------------------------------------------------------------------
# strict JSON
# --------------------------------------------------------------------------------------
def _clip(text: str, limit: int) -> str:
    """text, or its first `limit` characters plus a count of the characters left out."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(+{len(text) - limit} chars)"


def _shown_key(key: Any) -> str:
    """A key as shown in refusal messages: at most MAX_SHOWN_KEY characters."""
    return _clip(str(key), MAX_SHOWN_KEY)


def _shown_keys(keys: Any) -> list[str]:
    """Sorted keys as shown in refusal messages: at most MAX_SHOWN_KEYS, each capped."""
    shown = sorted(_shown_key(k) for k in keys)
    if len(shown) > MAX_SHOWN_KEYS:
        shown = shown[:MAX_SHOWN_KEYS] + [f"...(+{len(shown) - MAX_SHOWN_KEYS} more)"]
    return shown


def _shown_path(path: str) -> str:
    """An input path as shown in refusal messages: at most MAX_SHOWN_PATH characters."""
    return _clip(path, MAX_SHOWN_PATH)


def _refusal_line(message: str) -> str:
    """The CLI refusal as one JSON line of at most MAX_SHOWN_ERROR bytes (ASCII, JSON-escaped).

    JSON escaping can grow one character to 12 bytes, so the message is clipped until the whole
    line fits; with a limit of 0 only the ASCII "...(+N chars)" note remains, so this ends.
    """
    limit = MAX_SHOWN_ERROR - 64
    while True:
        line = json.dumps({"error": _clip(message, limit), "ok": False}, sort_keys=True)
        if len(line) <= MAX_SHOWN_ERROR:
            return line
        limit //= 2


def _pairs_hook(error_cls):
    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise error_cls(f"duplicate JSON key {_shown_key(key)!r} is not accepted")
            out[key] = value
        return out

    return hook


def loads_strict(text: str, error_cls=ModelArtifactError, *, allow_nonfinite: bool = False):
    """json.loads with duplicate-key refusal; NaN/Infinity refused unless allow_nonfinite."""

    def refuse_constant(name):
        raise error_cls(f"non-finite JSON constant {name} is not accepted")

    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_hook(error_cls),
            parse_constant=None if allow_nonfinite else refuse_constant,
        )
    except OperationalModelError:
        raise
    except (ValueError, RecursionError) as exc:
        raise error_cls(f"unable to parse JSON: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def kernel_file_sha256() -> str:
    """sha256 of this kernel's own source file, as it is on disk now."""
    try:
        return sha256_bytes(Path(__file__).resolve().read_bytes())
    except (NameError, OSError) as exc:
        raise ModelArtifactError(f"unable to hash the kernel's own source file: {exc}") from exc


# --------------------------------------------------------------------------------------
# input screening and normalization
# --------------------------------------------------------------------------------------
def normalized_key(key: Any) -> str:
    """v1 key normalization: strip, lower-case, '-' and ' ' become '_'."""
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


_FOLD_SEPARATORS = re.compile(r"[\s\-]+")


def folded_key(key: Any) -> str:
    """Unicode-folded key: NFKC, format characters (category Cf: zero-width space/joiner, soft
    hyphen, BOM, ...) removed, casefolded, whitespace/hyphen runs become '_', edges stripped.
    Full-width letters fold to ASCII.  Cross-script homoglyphs (for example Cyrillic letters)
    do not fold; such keys are still refused by the exact eight-field schema check."""
    text = unicodedata.normalize("NFKC", str(key))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return _FOLD_SEPARATORS.sub("_", text.casefold().strip())


def prohibited_match(key: Any) -> str | None:
    """Return the kind of match ('exact', 'token', 'substring', or one of these with
    ', unicode-folded') or None."""
    text = str(key)
    if len(text) <= MAX_SHOWN_KEY:
        return _prohibited_match_cached(text)
    return _prohibited_match_text(text)


def _prohibited_match_text(text: str) -> str | None:
    match = _prohibited_match_normalized(normalized_key(text))
    if match is not None:
        return match
    folded = _prohibited_match_normalized(folded_key(text))
    return None if folded is None else f"{folded}, unicode-folded"


_prohibited_match_cached = lru_cache(maxsize=4096)(_prohibited_match_text)


def _prohibited_match_normalized(norm: str) -> str | None:
    if norm in V1_PROHIBITED_KEYS:
        return "exact"
    if any(token in PROHIBITED_TOKENS for token in norm.split("_") if token):
        return "token"
    if any(fragment in norm for fragment in PROHIBITED_SUBSTRINGS):
        return "substring"
    return None


def screen_input(value: Any, path: str = "input", depth: int = 0) -> None:
    """Recursive screen, run before any schema check.

    Refuses prohibited key names at every level (all keys of a level are checked before any
    child is visited), text values anywhere, and nesting deeper than MAX_INPUT_DEPTH.
    """
    if depth > MAX_INPUT_DEPTH:
        raise ModelInputError(
            f"input nesting exceeds {MAX_INPUT_DEPTH} levels at {_shown_path(path)}"
        )
    if isinstance(value, Mapping):
        for key in value:
            match = prohibited_match(key)
            if match is not None:
                raise ModelInputError(
                    f"prohibited non-operational field at {_shown_path(path)}.{_shown_key(key)} "
                    f"({match} match)"
                )
        for key, child in value.items():
            screen_input(child, f"{path}.{_shown_key(key)}", depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            screen_input(child, f"{path}[{index}]", depth + 1)
    elif isinstance(value, (str, bytes, bytearray)):
        raise ModelInputError(
            f"text values are not accepted at {_shown_path(path)}; input is numeric only"
        )


def _normalize_value(name: str, minimum: float, maximum: float, kind: str, value: Any) -> float:
    if kind == "binary":
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, int) and value in (0, 1):
            return float(value)
        raise ModelInputError(f"{name} must be boolean or integer 0/1")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelInputError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ModelInputError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ModelInputError(f"{name} must be finite")
    if number < minimum or number > maximum:
        raise ModelInputError(f"{name} must be between {minimum:g} and {maximum:g}")
    return (number - minimum) / (maximum - minimum) + 0.0  # + 0.0 maps -0.0 to +0.0


def normalize_features(features: Any) -> list[float]:
    """Screen, schema-check and normalize the eight features (v1 arithmetic)."""
    if not isinstance(features, Mapping):
        raise ModelInputError("features must be a JSON object")
    screen_input(features, "input.features")
    keys = set(features)
    expected = set(FEATURE_NAMES)
    if keys != expected:
        details = []
        missing = sorted(expected - keys)
        unexpected = _shown_keys(keys - expected)
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ModelInputError("feature schema mismatch: " + ", ".join(details))
    return [
        _normalize_value(name, minimum, maximum, kind, features[name])
        for name, minimum, maximum, kind in FEATURE_SPECS
    ]


def unwrap_payload(payload: Any) -> Mapping:
    """Accept only {"features": {...}}; the screen runs over the whole payload first."""
    if not isinstance(payload, Mapping):
        raise ModelInputError('input must be a JSON object of the form {"features": {...}}')
    screen_input(payload)
    if "features" not in payload:
        raise ModelInputError('input must be wrapped as {"features": {...}}')
    if set(payload) != {"features"}:
        raise ModelInputError("wrapped inference input may contain only the features field")
    features = payload["features"]
    if not isinstance(features, Mapping):
        raise ModelInputError("features must be a JSON object")
    return features


# --------------------------------------------------------------------------------------
# scoring arithmetic (single source of truth; research code imports these)
# --------------------------------------------------------------------------------------
def sigmoid(value: float) -> float:
    """Numerically stable logistic, identical to v1."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def logit_clipped(score: float) -> float:
    """logit(clip(score, 1e-15, 1 - 1e-15)), the Platt input."""
    p = min(max(score, PLATT_CLIP), 1.0 - PLATT_CLIP)
    return math.log(p) - math.log1p(-p)


def isotonic_key(raw: float) -> float:
    """Isotonic input resolution: the raw score rounded to 12 decimals."""
    return round(raw, 12)


def apply_calibrator(calibrator: Mapping, raw: float) -> float:
    kind = calibrator["kind"]
    params = calibrator["params"]
    if kind == "none":
        return raw
    if kind == "platt":
        return sigmoid(params["a"] * logit_clipped(raw) + params["b"])
    if kind == "isotonic":
        index = bisect.bisect_right(params["thresholds"], isotonic_key(raw)) - 1
        return params["values"][max(index, 0)]
    raise ModelArtifactError("unknown calibrator kind")


def nonconformity(p1: float, label: int) -> float:
    """s(x, y) = 1 - p_y(x) with p_1 = p1 and p_0 = 1 - p1, so s(x, 0) is exactly p1."""
    return 1.0 - p1 if label == 1 else p1


def conformal_set(conformal: Mapping, p1: float) -> list[int]:
    q_hat = conformal["q_hat"]
    return [y for y in (0, 1) if nonconformity(p1, y) <= q_hat[str(y)]]


def advisory_from_set(prediction_set: Sequence[int]) -> tuple[str, str | None]:
    labels = list(prediction_set)
    if labels == [1]:
        return "ALERT", None
    if labels == [0]:
        return "NO_ALERT", None
    return "ABSTAIN", ("BOTH" if len(labels) == 2 else "EMPTY")


# --------------------------------------------------------------------------------------
# artifact and receipt validation
# --------------------------------------------------------------------------------------
def canonical_feature_manifest() -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "maximum": maximum,
            "minimum": minimum,
            "name": name,
            "transform": "identity" if kind == "binary" else "min_max",
        }
        for name, minimum, maximum, kind in FEATURE_SPECS
    ]


def _exact_keys(obj: Any, keys: frozenset, label: str) -> Mapping:
    if not isinstance(obj, Mapping):
        raise ModelArtifactError(f"{label} must be a JSON object")
    if set(obj) != set(keys):
        missing = sorted(set(keys) - set(obj))
        unexpected = _shown_keys(set(obj) - set(keys))
        raise ModelArtifactError(
            f"{label} fields mismatch: missing={missing}, unexpected={unexpected}"
        )
    return obj


def _finite(value: Any, label: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelArtifactError(f"{label} must be numeric")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ModelArtifactError(f"{label} is not a bounded finite number") from exc
    if not math.isfinite(number) or number < lo or number > hi:
        raise ModelArtifactError(f"{label} is not a bounded finite number in [{lo:g}, {hi:g}]")
    return number


def _integer(value: Any, label: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ModelArtifactError(f"{label} must be an integer in [{lo}, {hi}]")
    return value


def _text(value: Any, label: str, max_len: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise ModelArtifactError(f"{label} must be a non-empty string of at most {max_len} chars")
    return value


def _hex(value: Any, label: str, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ModelArtifactError(f"{label} is not a valid lower-case hex digest")
    return value


def validate_calibrator(obj: Any) -> dict:
    cal = _exact_keys(obj, frozenset({"kind", "params"}), "calibrator")
    kind = cal["kind"]
    params = cal["params"]
    if kind not in CALIBRATOR_KINDS:
        raise ModelArtifactError(
            f"unsupported calibrator kind (not shown); expected one of {list(CALIBRATOR_KINDS)}"
        )
    if kind == "none":
        _exact_keys(params, frozenset(), "calibrator.params")
        return {"kind": "none", "params": {}}
    if kind == "platt":
        _exact_keys(params, frozenset({"a", "b"}), "calibrator.params")
        return {
            "kind": "platt",
            "params": {
                "a": _finite(params["a"], "calibrator.a", -PARAM_LIMIT, PARAM_LIMIT),
                "b": _finite(params["b"], "calibrator.b", -PARAM_LIMIT, PARAM_LIMIT),
            },
        }
    _exact_keys(params, frozenset({"thresholds", "values"}), "calibrator.params")
    thresholds, values = params["thresholds"], params["values"]
    if (
        not isinstance(thresholds, list)
        or not isinstance(values, list)
        or not 1 <= len(thresholds) <= MAX_ISOTONIC_BLOCKS
        or len(thresholds) != len(values)
    ):
        raise ModelArtifactError("isotonic thresholds/values must be equal-length non-empty lists")
    t = [_finite(v, "isotonic threshold", 0.0, 1.0) for v in thresholds]
    v = [_finite(x, "isotonic value", 0.0, 1.0) for x in values]
    if any(b <= a for a, b in zip(t, t[1:])):
        raise ModelArtifactError("isotonic thresholds must be strictly increasing")
    if any(b < a for a, b in zip(v, v[1:])):
        raise ModelArtifactError("isotonic values must be non-decreasing")
    return {"kind": "isotonic", "params": {"thresholds": t, "values": v}}


def validate_conformal(obj: Any) -> dict:
    conf = _exact_keys(obj, CONFORMAL_KEYS, "conformal")
    if conf["method"] != CONFORMAL_METHOD:
        raise ModelArtifactError("unsupported conformal method")
    alpha = _finite(conf["alpha"], "conformal.alpha", 0.0, 1.0)
    if not 0.0 < alpha < 1.0:
        raise ModelArtifactError("conformal.alpha must be strictly between zero and one")
    q_hat = _exact_keys(conf["q_hat"], frozenset({"0", "1"}), "conformal.q_hat")
    n_cal = _exact_keys(conf["n_calibration"], frozenset({"0", "1"}), "conformal.n_calibration")
    return {
        "method": CONFORMAL_METHOD,
        "alpha": alpha,
        "q_hat": {k: _finite(q_hat[k], f"conformal.q_hat.{k}", 0.0, 1.0) for k in ("0", "1")},
        "n_calibration": {
            k: _integer(n_cal[k], f"conformal.n_calibration.{k}", 1, 10**8) for k in ("0", "1")
        },
    }


def validate_training(obj: Any) -> dict:
    tr = _exact_keys(obj, TRAINING_KEYS, "training")
    if tr["algorithm"] != TRAINING_ALGORITHM:
        raise ModelArtifactError("training algorithm mismatch")
    if tr["threshold_selection"] != THRESHOLD_SELECTION or tr["threshold_grid"] != THRESHOLD_GRID:
        raise ModelArtifactError("threshold selection provenance mismatch")
    if (
        not isinstance(tr["stop_reason"], str)
        or tr["stop_reason"] not in STOP_REASONS
        or not isinstance(tr["converged"], bool)
    ):
        raise ModelArtifactError("training stop provenance invalid")
    if tr["converged"] != (tr["stop_reason"] == "CONVERGED"):
        raise ModelArtifactError("training converged flag contradicts stop_reason")
    if _integer(tr["max_iter"], "training.max_iter", 1, 10**6) != MAX_ITER:
        raise ModelArtifactError("training.max_iter must be the registered 100")
    if _finite(tr["tol"], "training.tol", 0.0, 1.0) != TOL:
        raise ModelArtifactError("training.tol must be the registered 1e-10")
    _integer(tr["iterations"], "training.iterations", 1, MAX_ITER)
    if _finite(tr["l2"], "training.l2", 0.0, 1.0) not in REGISTERED_L2_GRID:
        raise ModelArtifactError("training.l2 must be on the registered grid {0, 1e-4, 1e-3, 1e-2}")
    splits = _exact_keys(tr["splits"], TRAINING_SPLIT_ROLES, "training.splits")
    for role in sorted(TRAINING_SPLIT_ROLES):
        entry = _exact_keys(splits[role], frozenset({"name", "rows", "sha256"}), f"training.splits.{role}")
        _text(entry["name"], f"training.splits.{role}.name", 64)
        _integer(entry["rows"], f"training.splits.{role}.rows", 1, 10**8)
        _hex(entry["sha256"], f"training.splits.{role}.sha256", _HEX64)
    return json.loads(json.dumps(tr))


def validate_generator(obj: Any) -> dict:
    gen = _exact_keys(obj, GENERATOR_KEYS, "generator")
    if gen["version"] != GENERATOR_VERSION:
        raise ModelArtifactError("generator version mismatch")
    _hex(gen["source_sha256"], "generator.source_sha256", _HEX64)
    _integer(gen["master_seed"], "generator.master_seed", 0, 2**63 - 1)
    _text(gen["row_schema"], "generator.row_schema", 128)
    return dict(gen)


def validate_artifact(artifact: Any) -> dict:
    """Validate a parsed model artifact; returns a new, normalized dict (input untouched)."""
    art = _exact_keys(artifact, ARTIFACT_KEYS, "model artifact")
    if art["schema"] != MODEL_SCHEMA:
        raise ModelArtifactError("model schema mismatch")
    if art["model_type"] != MODEL_TYPE:
        raise ModelArtifactError("unsupported model type")
    if art["purpose"] != MODEL_PURPOSE or art["authority"] != MODEL_AUTHORITY:
        raise ModelArtifactError("model truth-boundary mismatch")
    if art["synthetic_training_data"] is not True:
        raise ModelArtifactError("only a synthetic-training artifact is accepted")
    if art["feature_manifest"] != canonical_feature_manifest():
        raise ModelArtifactError("model feature manifest mismatch")
    if art["score_semantics"] != SCORE_SEMANTICS:
        raise ModelArtifactError("score semantics mismatch")
    weights = _exact_keys(art["weights"], frozenset(FEATURE_NAMES), "weights")
    threshold = _finite(art["decision_threshold"], "decision_threshold", 0.0, 1.0)
    grid_index = round(threshold * 100)
    if not 1 <= grid_index <= 99 or threshold != grid_index / 100:
        raise ModelArtifactError("decision_threshold must lie on the registered 0.01..0.99 grid")
    conformal = validate_conformal(art["conformal"])
    if conformal["alpha"] != REGISTERED_ALPHA:
        raise ModelArtifactError("conformal.alpha must be the registered 0.10")
    return {
        "schema": MODEL_SCHEMA,
        "model_type": MODEL_TYPE,
        "purpose": MODEL_PURPOSE,
        "authority": MODEL_AUTHORITY,
        "synthetic_training_data": True,
        "feature_manifest": canonical_feature_manifest(),
        "intercept": _finite(art["intercept"], "intercept", -PARAM_LIMIT, PARAM_LIMIT),
        "weights": {
            name: _finite(weights[name], f"weight.{name}", -PARAM_LIMIT, PARAM_LIMIT)
            for name in FEATURE_NAMES
        },
        "calibrator": validate_calibrator(art["calibrator"]),
        "decision_threshold": threshold,
        "conformal": conformal,
        "score_semantics": SCORE_SEMANTICS,
        "training": validate_training(art["training"]),
        "generator": validate_generator(art["generator"]),
    }


def validate_receipt(receipt: Any) -> dict:
    """Schema, truth boundary, digest formats and origin (commit, clean tree) of a receipt."""
    if not isinstance(receipt, Mapping) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise ModelArtifactError("artifact receipt schema mismatch")
    if receipt.get("purpose") != MODEL_PURPOSE or receipt.get("authority") != MODEL_AUTHORITY:
        raise ModelArtifactError("artifact receipt truth-boundary mismatch")
    rec = _exact_keys(receipt, RECEIPT_KEYS, "artifact receipt")
    if rec["attests"] != RECEIPT_ATTESTS:
        raise ModelArtifactError("artifact receipt attestation text mismatch")
    for key in (
        "model_sha256", "kernel_sha256", "generator_sha256", "data_manifest_sha256",
        "preregistration_sha256",
    ):
        _hex(rec[key], f"receipt.{key}", _HEX64)
    if rec["preregistration_sha256"] != PREREGISTRATION_SHA256:
        raise ModelArtifactError("receipt preregistration digest is not the registered protocol")
    source = _exact_keys(rec["source"], SOURCE_KEYS, "receipt.source")
    _text(source["repository"], "receipt.source.repository")
    if not isinstance(source["commit"], str) or not _HEX40.fullmatch(source["commit"]):
        raise ModelArtifactError("receipt source commit is not a 40-hex lower-case git commit")
    if source["tree_clean"] is not True:
        raise ModelArtifactError("receipt source tree was not clean at freeze; refusing")
    if source["base_v1_source"] != V1_BASE_SOURCE or source["v1_hub_revision"] != V1_HUB_REVISION:
        raise ModelArtifactError("receipt v1 base-source record mismatch")
    selection = _exact_keys(rec["selection"], SELECTION_KEYS, "receipt.selection")
    _text(selection["procedure"], "receipt.selection.procedure")
    _text(selection["selection_split"], "receipt.selection.selection_split", 64)
    _integer(selection["trials"], "receipt.selection.trials", 0, 10**6)
    for key in ("selected_trial", "stop_reason"):
        if selection[key] is not None:
            _text(selection[key], f"receipt.selection.{key}", 128)
    for key in SELECTION_DIGEST_KEYS:
        if selection[key] is not None:
            _hex(selection[key], f"receipt.selection.{key}", _HEX64)
    if selection["conformal_keep"] is not None and not isinstance(selection["conformal_keep"], bool):
        raise ModelArtifactError("receipt.selection.conformal_keep must be true, false or null")
    return json.loads(json.dumps(rec))


# --------------------------------------------------------------------------------------
# scorer and kernel
# --------------------------------------------------------------------------------------
class ArtifactScorer:
    """Scores with a validated artifact.  No receipt check: research use only.

    The CLI and OpsHealthKernel always verify the receipt first.
    """

    def __init__(self, artifact: Mapping):
        self._artifact = validate_artifact(artifact)
        self._weights = [self._artifact["weights"][name] for name in FEATURE_NAMES]

    @property
    def artifact(self) -> dict:
        return json.loads(json.dumps(self._artifact))

    def evaluate(self, normalized: Sequence[float]) -> dict[str, Any]:
        """Unrounded evaluation of one normalized vector (v1 operation order)."""
        linear = self._artifact["intercept"]
        contributions = []
        for weight, value in zip(self._weights, normalized, strict=True):
            contribution = weight * value
            contributions.append(contribution)
            linear += contribution
        raw = sigmoid(linear)
        calibrated = apply_calibrator(self._artifact["calibrator"], raw)
        prediction_set = conformal_set(self._artifact["conformal"], calibrated)
        advisory, reason = advisory_from_set(prediction_set)
        return {
            "linear": linear,
            "raw_score": raw,
            "calibrated_score": calibrated,
            "threshold_decision": calibrated >= self._artifact["decision_threshold"],
            "prediction_set": prediction_set,
            "advisory": advisory,
            "abstain_reason": reason,
            "contributions": contributions,
        }

    def coverage_statement(self) -> str:
        alpha = self._artifact["conformal"]["alpha"]
        return (
            f"Mondrian split-conformal set at alpha={alpha:g}: per-class coverage of at least "
            f"{1.0 - alpha:g} holds only for data exchangeable with the synthetic nominal "
            "conformal split; under distribution shift coverage is descriptive only. SYNTHETIC."
        )

    def advise(self, payload: Any) -> dict[str, Any]:
        """Score one wrapped payload {"features": {...}} and return the v2 advisory."""
        normalized = normalize_features(unwrap_payload(payload))
        result = self.evaluate(normalized)
        return {
            "schema": ADVISORY_SCHEMA,
            "advisory": result["advisory"],
            "abstain_reason": result["abstain_reason"],
            "prediction_set": result["prediction_set"],
            "operator_attention_score": round(result["calibrated_score"], 12),
            "threshold_decision": result["threshold_decision"],
            "decision_threshold": self._artifact["decision_threshold"],
            "score_semantics": SCORE_SEMANTICS,
            "purpose": MODEL_PURPOSE,
            "authority": dict(AUTHORITY_MAP),
            "normalized_feature_contributions": {
                name: round(c, 12) + 0.0  # + 0.0: no negative zero in the report
                for name, c in zip(FEATURE_NAMES, result["contributions"], strict=True)
            },
            "coverage_statement": self.coverage_statement(),
        }


class OpsHealthKernel(ArtifactScorer):
    """Verify the receipt (including this file's own sha256), then load the artifact."""

    def __init__(self, model_path: Path | str, receipt_path: Path | str):
        model_path = Path(model_path).resolve()
        receipt_path = Path(receipt_path).resolve()
        try:
            model_bytes = model_path.read_bytes()
        except OSError as exc:
            raise ModelArtifactError(f"unable to read model artifact: {exc}") from exc
        try:
            receipt_text = receipt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ModelArtifactError(f"unable to read artifact receipt: {exc}") from exc
        receipt = validate_receipt(loads_strict(receipt_text, ModelArtifactError))
        if kernel_file_sha256() != receipt["kernel_sha256"]:
            raise ModelArtifactError("kernel sha256 does not match receipt")
        if sha256_bytes(model_bytes) != receipt["model_sha256"]:
            raise ModelArtifactError("model artifact hash does not match receipt")
        try:
            artifact = loads_strict(model_bytes.decode("utf-8"), ModelArtifactError)
        except UnicodeDecodeError as exc:
            raise ModelArtifactError(f"unable to decode model artifact: {exc}") from exc
        super().__init__(artifact)
        if self._artifact["generator"]["source_sha256"] != receipt["generator_sha256"]:
            raise ModelArtifactError("generator sha256 differs between model and receipt")
        self.receipt = receipt


def read_cli_input(path: str) -> Any:
    try:
        raw = sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()
    except OSError as exc:
        raise ModelInputError(f"unable to read input JSON: {exc}") from exc
    if len(raw) > MAX_INPUT_BYTES:
        raise ModelInputError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelInputError(f"unable to read input JSON: {exc}") from exc
    return loads_strict(text, ModelInputError, allow_nonfinite=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score synthetic operational transport telemetry (advisory only, SYNTHETIC)."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--input", required=True, help="JSON file, or - for stdin")
    args = parser.parse_args(argv)
    try:
        kernel = OpsHealthKernel(args.model, args.receipt)
        advisory = kernel.advise(read_cli_input(args.input))
    except OperationalModelError as exc:
        print(_refusal_line(str(exc)), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 -- fail closed: refuse, never print a stack trace
        error = f"internal error ({type(exc).__name__}); refusing"
        print(json.dumps({"ok": False, "error": error}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(advisory, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
