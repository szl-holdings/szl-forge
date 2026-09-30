from __future__ import annotations

"""Fail-closed inference for the synthetic transport-health advisory model.

The model in this module is deliberately isolated from result ingestion.  It
accepts only a fixed set of operational transport counters and produces an
operator-attention advisory.  It cannot acknowledge messages, change gateway
state, interpret results, or authorize clinical use.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import hashlib
import json
import math
import sys


MODEL_SCHEMA = "szl-oac/transport-health-logistic-model/v1"
RECEIPT_SCHEMA = "szl-oac/transport-health-artifact-receipt/v1"
ADVISORY_SCHEMA = "szl-oac/transport-health-advisory/v1"
MODEL_PURPOSE = "synthetic_operational_transport_attention_only"
MODEL_AUTHORITY = "advisory_only_no_ack_release_or_clinical_authority"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    minimum: float
    maximum: float
    kind: str

    def normalize(self, value: Any) -> float:
        if self.kind == "binary":
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, int) and value in (0, 1):
                return float(value)
            raise ModelInputError(f"{self.name} must be boolean or integer 0/1")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelInputError(f"{self.name} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ModelInputError(f"{self.name} must be finite")
        if number < self.minimum or number > self.maximum:
            raise ModelInputError(
                f"{self.name} must be between {self.minimum:g} and {self.maximum:g}"
            )
        return (number - self.minimum) / (self.maximum - self.minimum)


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("listener_running", 0.0, 1.0, "binary"),
    FeatureSpec("tls_enabled", 0.0, 1.0, "binary"),
    FeatureSpec("peer_allowlist_configured", 0.0, 1.0, "binary"),
    FeatureSpec("queue_utilization", 0.0, 1.0, "continuous"),
    FeatureSpec("consecutive_failures", 0.0, 20.0, "continuous"),
    FeatureSpec("seconds_since_last_success", 0.0, 86400.0, "continuous"),
    FeatureSpec("ledger_integrity_ok", 0.0, 1.0, "binary"),
    FeatureSpec("configuration_valid", 0.0, 1.0, "binary"),
)
FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)

_PROHIBITED_KEYS = frozenset(
    {
        "address",
        "date_of_birth",
        "dob",
        "email",
        "fhir",
        "hl7",
        "mrn",
        "name",
        "obx",
        "order",
        "order_id",
        "patient",
        "patient_id",
        "phone",
        "pid",
        "result",
        "result_value",
        "specimen",
        "specimen_id",
        "ssn",
    }
)


class OperationalModelError(ValueError):
    """Base error for fail-closed artifact and input validation."""


class ModelArtifactError(OperationalModelError):
    """Raised when a model or receipt fails integrity/schema validation."""


class ModelInputError(OperationalModelError):
    """Raised when inference input is outside the operational-only schema."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _reject_sensitive_keys(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in _PROHIBITED_KEYS:
                raise ModelInputError(f"prohibited non-operational field at {path}.{key}")
            _reject_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, f"{path}[{index}]")
    elif isinstance(value, str):
        upper = value.upper()
        if upper.startswith("MSH|") or "\rPID|" in upper or "\rOBX|" in upper:
            raise ModelInputError(f"raw HL7-like content is prohibited at {path}")


def normalize_features(features: Mapping[str, Any]) -> list[float]:
    if not isinstance(features, Mapping):
        raise ModelInputError("features must be a JSON object")
    _reject_sensitive_keys(features)
    keys = set(features)
    expected = set(FEATURE_NAMES)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise ModelInputError("feature schema mismatch: " + ", ".join(details))
    return [spec.normalize(features[spec.name]) for spec in FEATURE_SPECS]


def _finite_number(value: Any, label: str, *, absolute_limit: float = 100.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelArtifactError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or abs(number) > absolute_limit:
        raise ModelArtifactError(f"{label} is not a bounded finite number")
    return number


def _canonical_feature_manifest() -> list[dict[str, Any]]:
    return [
        {
            "kind": spec.kind,
            "maximum": spec.maximum,
            "minimum": spec.minimum,
            "name": spec.name,
            "transform": "identity" if spec.kind == "binary" else "min_max",
        }
        for spec in FEATURE_SPECS
    ]


class OperationalHealthKernel:
    """Load a verified artifact and emit a non-authoritative health advisory."""

    def __init__(self, model_path: Path | str, receipt_path: Path | str):
        self.model_path = Path(model_path).resolve()
        self.receipt_path = Path(receipt_path).resolve()
        try:
            model_bytes = self.model_path.read_bytes()
        except OSError as exc:
            raise ModelArtifactError(f"unable to read model artifact: {exc}") from exc
        self._verify_receipt(model_bytes)
        try:
            artifact = json.loads(model_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"unable to decode model artifact: {exc}") from exc
        self._artifact = self._validate_artifact(artifact)

    def _verify_receipt(self, model_bytes: bytes) -> None:
        try:
            receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"unable to read artifact receipt: {exc}") from exc
        if not isinstance(receipt, Mapping) or receipt.get("schema") != RECEIPT_SCHEMA:
            raise ModelArtifactError("artifact receipt schema mismatch")
        if receipt.get("purpose") != MODEL_PURPOSE or receipt.get("authority") != MODEL_AUTHORITY:
            raise ModelArtifactError("artifact receipt truth-boundary mismatch")
        expected = receipt.get("model_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ModelArtifactError("artifact receipt has no valid model_sha256")
        observed = hashlib.sha256(model_bytes).hexdigest()
        if observed != expected:
            raise ModelArtifactError("model artifact hash does not match receipt")

    @staticmethod
    def _validate_artifact(artifact: Any) -> dict[str, Any]:
        if not isinstance(artifact, dict):
            raise ModelArtifactError("model artifact must be a JSON object")
        required = {
            "schema",
            "model_type",
            "purpose",
            "authority",
            "synthetic_training_data",
            "feature_manifest",
            "intercept",
            "weights",
            "decision_threshold",
            "score_semantics",
            "training",
        }
        if set(artifact) != required:
            raise ModelArtifactError("model artifact fields do not match the v1 schema")
        if artifact.get("schema") != MODEL_SCHEMA:
            raise ModelArtifactError("model schema mismatch")
        if artifact.get("model_type") != "binary_logistic_regression_standard_library":
            raise ModelArtifactError("unsupported model type")
        if artifact.get("purpose") != MODEL_PURPOSE or artifact.get("authority") != MODEL_AUTHORITY:
            raise ModelArtifactError("model truth-boundary mismatch")
        if artifact.get("synthetic_training_data") is not True:
            raise ModelArtifactError("only a synthetic-training artifact is accepted")
        if artifact.get("feature_manifest") != _canonical_feature_manifest():
            raise ModelArtifactError("model feature manifest mismatch")
        weights = artifact.get("weights")
        if not isinstance(weights, dict) or set(weights) != set(FEATURE_NAMES):
            raise ModelArtifactError("model weight schema mismatch")
        artifact["intercept"] = _finite_number(artifact.get("intercept"), "intercept")
        artifact["weights"] = {
            name: _finite_number(weights[name], f"weight.{name}") for name in FEATURE_NAMES
        }
        threshold = _finite_number(
            artifact.get("decision_threshold"), "decision_threshold", absolute_limit=1.0
        )
        if threshold <= 0.0 or threshold >= 1.0:
            raise ModelArtifactError("decision_threshold must be between zero and one")
        artifact["decision_threshold"] = threshold
        if artifact.get("score_semantics") != "synthetic_attention_score_not_production_calibrated":
            raise ModelArtifactError("score semantics mismatch")
        training = artifact.get("training")
        if not isinstance(training, dict):
            raise ModelArtifactError("training provenance must be an object")
        expected_training = {
            "algorithm": "batch_gradient_descent_logistic_regression",
            "epochs": 2400,
            "l2_penalty": 0.01,
            "learning_rate": 0.42,
            "seed": 2500,
            "threshold_selection": "validation_balanced_accuracy_then_f1",
            "training_rows": 768,
        }
        if training != expected_training:
            raise ModelArtifactError("training provenance mismatch")
        return artifact

    def score(self, features: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_features(features)
        linear = self._artifact["intercept"]
        contributions: dict[str, float] = {}
        for name, value in zip(FEATURE_NAMES, normalized, strict=True):
            contribution = self._artifact["weights"][name] * value
            contributions[name] = round(contribution, 12)
            linear += contribution
        if linear >= 0.0:
            score = 1.0 / (1.0 + math.exp(-min(linear, 700.0)))
        else:
            exp_value = math.exp(max(linear, -700.0))
            score = exp_value / (1.0 + exp_value)
        threshold = self._artifact["decision_threshold"]
        return {
            "schema": ADVISORY_SCHEMA,
            "operator_attention_required": score >= threshold,
            "operator_attention_score": round(score, 12),
            "decision_threshold": threshold,
            "score_semantics": "synthetic_attention_score_not_production_calibrated",
            "purpose": MODEL_PURPOSE,
            "authority": {
                "acknowledgement": False,
                "clinical_decision": False,
                "device_control": False,
                "result_interpretation": False,
                "result_release": False,
            },
            "normalized_feature_contributions": contributions,
        }


def _read_cli_input(path: str) -> Mapping[str, Any]:
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelInputError(f"unable to read input JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ModelInputError("input JSON must be an object")
    _reject_sensitive_keys(parsed)
    if "features" in parsed:
        if set(parsed) != {"features"}:
            raise ModelInputError("wrapped inference input may contain only the features field")
        parsed = parsed["features"]
    if not isinstance(parsed, Mapping):
        raise ModelInputError("features must be an object")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score synthetic operational transport telemetry (advisory only)."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--input", required=True, help="JSON file, or - for stdin")
    args = parser.parse_args(argv)
    try:
        kernel = OperationalHealthKernel(args.model, args.receipt)
        advisory = kernel.score(_read_cli_input(args.input))
    except OperationalModelError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(advisory, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
