from __future__ import annotations

"""Reproducibly generate and train the synthetic transport-health model.

This script intentionally uses only the Python standard library.  It generates
operational transport counters, never patient/specimen/order/result data.  The
fixed seed, stable JSON encoding, receipts, and verify mode make every committed
artifact byte-reproducible.
"""

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import argparse
import json
import math
import random
import shutil
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oac_operational_health import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_SPECS,
    MODEL_AUTHORITY,
    MODEL_PURPOSE,
    MODEL_SCHEMA,
    RECEIPT_SCHEMA,
    normalize_features,
    sha256_file,
)


SEED = 2500
SPLIT_ROWS = {"train": 768, "validation": 192, "test": 240}
DATASET_SCHEMA = "szl-oac/transport-health-observation/v1"
DATASET_RECEIPT_SCHEMA = "szl-oac/transport-health-dataset-receipt/v1"
DATASET_PURPOSE = "synthetic_operational_transport_observability_only"
GENERATOR_RELATIVE_PATH = "tools/train_operational_health_model.py"
KERNEL_RELATIVE_PATH = "src/oac_operational_health.py"
MODEL_STAGE_RELATIVE = Path("huggingface/model/oac-clinical-transport-health-v1")
DATASET_STAGE_RELATIVE = Path(
    "huggingface/dataset/oac-clinical-transport-observability-synthetic"
)
MODEL_STAGE_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "artifact_receipt.json",
        "example_input.json",
        "model.json",
        "oac_operational_health.py",
    }
)
DATASET_STAGE_FILES = frozenset(
    {
        "LICENSE",
        "README.md",
        "data/test.jsonl",
        "data/train.jsonl",
        "data/validation.jsonl",
        "dataset_receipt.json",
        "schema.json",
        "training_source_snapshot.py",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _weighted_failure_count(rng: random.Random) -> int:
    draw = rng.random()
    if draw < 0.72:
        return 0
    if draw < 0.86:
        return 1
    if draw < 0.93:
        return 2
    if draw < 0.97:
        return rng.randint(3, 5)
    return rng.randint(6, 20)


def _generate_features(rng: random.Random) -> dict[str, int | float]:
    return {
        "listener_running": int(rng.random() < 0.94),
        "tls_enabled": int(rng.random() < 0.90),
        "peer_allowlist_configured": int(rng.random() < 0.93),
        "queue_utilization": round(min(1.0, rng.betavariate(1.25, 4.5)), 6),
        "consecutive_failures": _weighted_failure_count(rng),
        "seconds_since_last_success": round(
            min(86400.0, rng.expovariate(1.0 / 900.0)), 6
        ),
        "ledger_integrity_ok": int(rng.random() < 0.985),
        "configuration_valid": int(rng.random() < 0.96),
    }


def _synthetic_label_probability(features: Mapping[str, Any]) -> float:
    values = dict(zip(FEATURE_NAMES, normalize_features(features), strict=True))
    linear = (
        -3.6
        + 3.1 * (1.0 - values["listener_running"])
        + 1.15 * (1.0 - values["tls_enabled"])
        + 1.35 * (1.0 - values["peer_allowlist_configured"])
        + 4.2 * values["queue_utilization"]
        + 2.7 * values["consecutive_failures"]
        + 2.2 * values["seconds_since_last_success"]
        + 4.4 * (1.0 - values["ledger_integrity_ok"])
        + 3.4 * (1.0 - values["configuration_valid"])
    )
    return 1.0 / (1.0 + math.exp(-linear))


def generate_dataset() -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(SEED)
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, row_count in SPLIT_ROWS.items():
        rows: list[dict[str, Any]] = []
        for index in range(row_count):
            features = _generate_features(rng)
            probability = _synthetic_label_probability(features)
            label = rng.random() < probability
            rows.append(
                {
                    "schema": DATASET_SCHEMA,
                    "synthetic": True,
                    "sample_id": f"{split}-{index:06d}",
                    "features": features,
                    "label": {"operator_attention_required": label},
                }
            )
        splits[split] = rows
    return splits


def _vectors(rows: Sequence[Mapping[str, Any]]) -> tuple[list[list[float]], list[int]]:
    x = [normalize_features(row["features"]) for row in rows]
    y = [int(bool(row["label"]["operator_attention_required"])) for row in rows]
    return x, y


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def train_logistic_regression(
    rows: Sequence[Mapping[str, Any]],
    *,
    epochs: int = 2400,
    learning_rate: float = 0.42,
    l2_penalty: float = 0.01,
) -> tuple[float, list[float]]:
    x, y = _vectors(rows)
    intercept = 0.0
    weights = [0.0] * len(FEATURE_NAMES)
    count = float(len(x))
    for _ in range(epochs):
        intercept_gradient = 0.0
        gradients = [0.0] * len(weights)
        for vector, target in zip(x, y, strict=True):
            probability = _sigmoid(
                intercept + sum(weight * value for weight, value in zip(weights, vector, strict=True))
            )
            error = probability - target
            intercept_gradient += error
            for index, value in enumerate(vector):
                gradients[index] += error * value
        intercept -= learning_rate * intercept_gradient / count
        for index in range(len(weights)):
            regularized = gradients[index] / count + l2_penalty * weights[index]
            weights[index] -= learning_rate * regularized
    return intercept, weights


def predict_probabilities(
    rows: Sequence[Mapping[str, Any]], intercept: float, weights: Sequence[float]
) -> list[float]:
    x, _ = _vectors(rows)
    return [
        _sigmoid(intercept + sum(w * value for w, value in zip(weights, vector, strict=True)))
        for vector in x
    ]


def _confusion(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, int]:
    counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
    for label, score in zip(labels, scores, strict=True):
        predicted = int(score >= threshold)
        if label == 1 and predicted == 1:
            counts["true_positive"] += 1
        elif label == 0 and predicted == 0:
            counts["true_negative"] += 1
        elif label == 0 and predicted == 1:
            counts["false_positive"] += 1
        else:
            counts["false_negative"] += 1
    return counts


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.0
    concordance = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                concordance += 1.0
            elif positive == negative:
                concordance += 0.5
    return concordance / (len(positives) * len(negatives))


def metrics(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    _, labels = _vectors(rows)
    counts = _confusion(labels, scores, threshold)
    tp = counts["true_positive"]
    tn = counts["true_negative"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    clipped = [min(max(score, 1e-15), 1.0 - 1e-15) for score in scores]
    log_loss = -sum(
        label * math.log(score) + (1 - label) * math.log(1.0 - score)
        for label, score in zip(labels, clipped, strict=True)
    ) / len(labels)
    return {
        "rows": len(labels),
        "positive_rows": sum(labels),
        "negative_rows": len(labels) - sum(labels),
        "accuracy": round(_safe_divide(tp + tn, len(labels)), 12),
        "balanced_accuracy": round((recall + specificity) / 2.0, 12),
        "precision": round(precision, 12),
        "recall": round(recall, 12),
        "specificity": round(specificity, 12),
        "f1": round(_safe_divide(2.0 * precision * recall, precision + recall), 12),
        "roc_auc": round(_roc_auc(labels, scores), 12),
        "log_loss": round(log_loss, 12),
        "confusion": counts,
    }


def choose_threshold(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> float:
    best: tuple[float, float, float] | None = None
    for step in range(5, 96):
        threshold = step / 100.0
        candidate_metrics = metrics(rows, scores, threshold)
        # Deterministic tie-break: balanced accuracy, F1, then proximity to 0.5.
        rank = (
            candidate_metrics["balanced_accuracy"],
            candidate_metrics["f1"],
            -abs(threshold - 0.5),
        )
        if best is None or rank > best:
            best = rank
            chosen = threshold
    return chosen


def dataset_json_schema() -> dict[str, Any]:
    feature_properties: dict[str, Any] = {}
    for spec in FEATURE_SPECS:
        feature_properties[spec.name] = {
            "type": "integer" if spec.kind == "binary" else "number",
            "minimum": spec.minimum,
            "maximum": spec.maximum,
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": DATASET_SCHEMA,
        "title": "Synthetic OAC transport-health observation",
        "description": "Operational-only synthetic telemetry; no PHI or clinical result fields.",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "synthetic", "sample_id", "features", "label"],
        "properties": {
            "schema": {"const": DATASET_SCHEMA},
            "synthetic": {"const": True},
            "sample_id": {"type": "string", "pattern": "^(train|validation|test)-[0-9]{6}$"},
            "features": {
                "type": "object",
                "additionalProperties": False,
                "required": list(FEATURE_NAMES),
                "properties": feature_properties,
            },
            "label": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operator_attention_required"],
                "properties": {"operator_attention_required": {"type": "boolean"}},
            },
        },
    }


def _relative_hashes(base: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {path.relative_to(base).as_posix(): sha256_file(path) for path in sorted(paths)}


def _stage_files(stage: Path) -> set[str]:
    if not stage.exists():
        return set()
    return {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}


def _reject_unexpected_stage_files(stage: Path, allowed: frozenset[str]) -> None:
    unexpected = sorted(_stage_files(stage) - allowed)
    if unexpected:
        raise RuntimeError(
            f"refusing to package unexpected files under {stage}: {unexpected}"
        )


def _copy_package_input(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        destination.write_bytes(source.read_text(encoding="utf-8").encode("utf-8"))


def build(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    canonical = output_root / "operational-model"
    data_dir = canonical / "data"
    artifact_dir = canonical / "artifacts"
    model_stage = output_root / MODEL_STAGE_RELATIVE
    dataset_stage = output_root / DATASET_STAGE_RELATIVE
    _reject_unexpected_stage_files(model_stage, MODEL_STAGE_FILES)
    _reject_unexpected_stage_files(dataset_stage, DATASET_STAGE_FILES)
    for directory in (data_dir, artifact_dir, model_stage, dataset_stage / "data"):
        directory.mkdir(parents=True, exist_ok=True)

    splits = generate_dataset()
    data_paths: list[Path] = []
    for split, rows in splits.items():
        path = data_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        data_paths.append(path)

    schema_path = canonical / "schema.json"
    write_json(schema_path, dataset_json_schema())

    intercept, weights = train_logistic_regression(splits["train"])
    validation_scores = predict_probabilities(splits["validation"], intercept, weights)
    threshold = choose_threshold(splits["validation"], validation_scores)
    split_metrics = {
        split: metrics(rows, predict_probabilities(rows, intercept, weights), threshold)
        for split, rows in splits.items()
    }

    artifact = {
        "schema": MODEL_SCHEMA,
        "model_type": "binary_logistic_regression_standard_library",
        "purpose": MODEL_PURPOSE,
        "authority": MODEL_AUTHORITY,
        "synthetic_training_data": True,
        "feature_manifest": [
            {
                "kind": spec.kind,
                "maximum": spec.maximum,
                "minimum": spec.minimum,
                "name": spec.name,
                "transform": "identity" if spec.kind == "binary" else "min_max",
            }
            for spec in FEATURE_SPECS
        ],
        "intercept": round(intercept, 12),
        "weights": {
            name: round(weight, 12) for name, weight in zip(FEATURE_NAMES, weights, strict=True)
        },
        "decision_threshold": threshold,
        "score_semantics": "synthetic_attention_score_not_production_calibrated",
        "training": {
            "algorithm": "batch_gradient_descent_logistic_regression",
            "epochs": 2400,
            "l2_penalty": 0.01,
            "learning_rate": 0.42,
            "seed": SEED,
            "threshold_selection": "validation_balanced_accuracy_then_f1",
            "training_rows": SPLIT_ROWS["train"],
        },
    }
    model_path = artifact_dir / "model.json"
    write_json(model_path, artifact)

    generator_path = PROJECT_ROOT / GENERATOR_RELATIVE_PATH
    kernel_path = PROJECT_ROOT / KERNEL_RELATIVE_PATH
    dataset_receipt = {
        "schema": DATASET_RECEIPT_SCHEMA,
        "purpose": DATASET_PURPOSE,
        "authority": MODEL_AUTHORITY,
        "contains_phi": False,
        "contains_clinical_results": False,
        "generated_data": "synthetic_only",
        "generator_sha256": sha256_file(generator_path),
        "schema_sha256": sha256_file(schema_path),
        "seed": SEED,
        "split_rows": SPLIT_ROWS,
        "files": _relative_hashes(canonical, data_paths),
    }
    dataset_receipt_path = artifact_dir / "dataset-receipt.json"
    write_json(dataset_receipt_path, dataset_receipt)

    model_receipt = {
        "schema": RECEIPT_SCHEMA,
        "purpose": MODEL_PURPOSE,
        "authority": MODEL_AUTHORITY,
        "model_sha256": sha256_file(model_path),
        "kernel_sha256": sha256_file(kernel_path),
        "generator_sha256": sha256_file(generator_path),
        "dataset_receipt_sha256": sha256_file(dataset_receipt_path),
        "seed": SEED,
        "metrics_scope": "fixed_seed_synthetic_splits_only_not_production_validation",
        "metrics": split_metrics,
    }
    model_receipt_path = artifact_dir / "model-receipt.json"
    write_json(model_receipt_path, model_receipt)

    example_input = {
        "features": {
            "listener_running": 1,
            "tls_enabled": 1,
            "peer_allowlist_configured": 1,
            "queue_utilization": 0.12,
            "consecutive_failures": 0,
            "seconds_since_last_success": 14.0,
            "ledger_integrity_ok": 1,
            "configuration_valid": 1,
        }
    }
    example_path = canonical / "example-input.json"
    write_json(example_path, example_input)

    shutil.copyfile(model_path, model_stage / "model.json")
    shutil.copyfile(model_receipt_path, model_stage / "artifact_receipt.json")
    shutil.copyfile(kernel_path, model_stage / "oac_operational_health.py")
    shutil.copyfile(example_path, model_stage / "example_input.json")
    _copy_package_input(PROJECT_ROOT / "LICENSE", model_stage / "LICENSE")
    _copy_package_input(PROJECT_ROOT / MODEL_STAGE_RELATIVE / "README.md", model_stage / "README.md")

    for data_path in data_paths:
        shutil.copyfile(data_path, dataset_stage / "data" / data_path.name)
    shutil.copyfile(schema_path, dataset_stage / "schema.json")
    shutil.copyfile(dataset_receipt_path, dataset_stage / "dataset_receipt.json")
    shutil.copyfile(generator_path, dataset_stage / "training_source_snapshot.py")
    _copy_package_input(PROJECT_ROOT / "LICENSE", dataset_stage / "LICENSE")
    _copy_package_input(
        PROJECT_ROOT / DATASET_STAGE_RELATIVE / "README.md", dataset_stage / "README.md"
    )

    if _stage_files(model_stage) != set(MODEL_STAGE_FILES):
        raise RuntimeError("model staging package manifest is incomplete")
    if _stage_files(dataset_stage) != set(DATASET_STAGE_FILES):
        raise RuntimeError("dataset staging package manifest is incomplete")

    return {
        "model_sha256": sha256_file(model_path),
        "dataset_receipt_sha256": sha256_file(dataset_receipt_path),
        "model_receipt_sha256": sha256_file(model_receipt_path),
        "threshold": threshold,
        "metrics": split_metrics,
        "split_rows": SPLIT_ROWS,
    }


def generated_relative_paths() -> list[Path]:
    paths = [
        Path("operational-model/schema.json"),
        Path("operational-model/example-input.json"),
        Path("operational-model/artifacts/model.json"),
        Path("operational-model/artifacts/model-receipt.json"),
        Path("operational-model/artifacts/dataset-receipt.json"),
        Path("huggingface/model/oac-clinical-transport-health-v1/model.json"),
        Path("huggingface/model/oac-clinical-transport-health-v1/artifact_receipt.json"),
        Path("huggingface/model/oac-clinical-transport-health-v1/oac_operational_health.py"),
        Path("huggingface/model/oac-clinical-transport-health-v1/example_input.json"),
        Path("huggingface/model/oac-clinical-transport-health-v1/LICENSE"),
        Path("huggingface/model/oac-clinical-transport-health-v1/README.md"),
        Path("huggingface/dataset/oac-clinical-transport-observability-synthetic/schema.json"),
        Path("huggingface/dataset/oac-clinical-transport-observability-synthetic/dataset_receipt.json"),
        Path("huggingface/dataset/oac-clinical-transport-observability-synthetic/training_source_snapshot.py"),
        Path("huggingface/dataset/oac-clinical-transport-observability-synthetic/LICENSE"),
        Path("huggingface/dataset/oac-clinical-transport-observability-synthetic/README.md"),
    ]
    for split in SPLIT_ROWS:
        paths.append(Path(f"operational-model/data/{split}.jsonl"))
        paths.append(
            Path(
                "huggingface/dataset/oac-clinical-transport-observability-synthetic"
                f"/data/{split}.jsonl"
            )
        )
    return sorted(paths)


def verify_committed() -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    committed_stages = (
        (PROJECT_ROOT / MODEL_STAGE_RELATIVE, MODEL_STAGE_FILES),
        (PROJECT_ROOT / DATASET_STAGE_RELATIVE, DATASET_STAGE_FILES),
    )
    for stage, expected in committed_stages:
        observed = _stage_files(stage)
        if observed != set(expected):
            mismatches.append(
                f"stage manifest mismatch at {stage.relative_to(PROJECT_ROOT).as_posix()}: "
                f"missing={sorted(set(expected) - observed)}, "
                f"unexpected={sorted(observed - set(expected))}"
            )
    with tempfile.TemporaryDirectory(prefix="oac-operational-model-") as temporary:
        generated_root = Path(temporary)
        build(generated_root)
        for relative in generated_relative_paths():
            committed = PROJECT_ROOT / relative
            regenerated = generated_root / relative
            if not committed.is_file():
                mismatches.append(f"missing committed artifact: {relative.as_posix()}")
            elif committed.read_bytes() != regenerated.read_bytes():
                mismatches.append(f"artifact drift: {relative.as_posix()}")
    return not mismatches, mismatches


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
        help="root for operational-model/ and huggingface/ outputs",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="regenerate in a temporary directory and byte-compare committed artifacts",
    )
    args = parser.parse_args(argv)
    if args.verify:
        ok, mismatches = verify_committed()
        print(json.dumps({"ok": ok, "mismatches": mismatches}, sort_keys=True))
        return 0 if ok else 1
    summary = build(args.output_root)
    print(json.dumps({"ok": True, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
