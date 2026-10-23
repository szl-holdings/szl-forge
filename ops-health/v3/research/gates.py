"""v3 gate set (protocol s.11.2, definitions frozen at registration). SYNTHETIC; standard library.

All gates are evaluated on the ``gates`` split (purpose GATES); the reference is the frozen v3.
A gate **catches** a fixture iff the fixture's value is strictly below the reference's own value
(as v2). An artifact that the kernel refuses to load scores 0 on every gate that needs kernel
output (fail closed): g_failclosed, g_monotone, g_validation, g_calibration, g_baseline_valid and
g_ba_rel. g_truth_sign, g_coef_band and g_threshold_band read the artifact's own parameters,
declared table and threshold index from its bytes (strict JSON); an artifact whose bytes or
table cannot be read scores 0 on them too.

Kept from v2 s.10 (v2's definitions, reused from ``v2/research/gates.py``):
  g_integrity       1 if the kernel accepts the model + receipt pair, else 0
  g_failclosed      fraction of the 76 probes refused: v2's 66 input probes (``FAILCLOSED_PROBES``
                    of v2, unchanged) and the transform-table probes TT1-TT10 of s.6.7
  g_truth_sign      sign agreement with generator truth, k/8
  g_monotone        v2's metamorphic checks (19,600 checks on v2's 4,000-point grid)
  g_validation      min(1, BA_gates / 0.70)
  g_calibration     1 if ECE_gates <= 0.05, else max(0, 1 - (ECE_gates - 0.05) / 0.10)
  g_baseline_valid  1 if the s.11.1 preconditions hold (with the 76 probes for (iv)), else 0
New, each in {0, 1}:
  g_coef_band       1 iff |theta_j(f) - theta_j(ref)| <= tol_j for the intercept and the eight
                    weights, each theta first mapped to representation R with the artifact's
                    declared table (``representation.artifact_theta_r``)
  g_threshold_band  1 iff |k(f) - k(ref)| <= tol_k, integers (grid index of s.6.4)
  g_ba_rel          1 iff BA_gates(f) >= BA_gates(ref) - tol_BA
Gate types (s.11.2): parameter space = g_coef_band, g_threshold_band, g_truth_sign; outputs =
g_ba_rel, g_validation, g_calibration, g_monotone.

Transform-table probes (s.6.7), each a modified copy of the fixture's own model.json presented
with a CONSISTENT receipt (model_sha256 regenerated, every other field unchanged), so the only
reason to refuse it is the artifact itself; each must be refused at load:
  TT1 seconds scale 86400; TT2 seconds scale 0; TT3 seconds scale -900; TT4 seconds scale NaN;
  TT5 seconds scale +inf; TT6 seconds offset 1; TT7 tls_enabled scale 2; TT8 the seconds entry
  missing; TT9 an extra entry for a field outside the eight; TT10 a duplicate key inside the
  table. A probe is refused iff loading it raises (any exception, as v2's input probes).

Tolerances are inputs here (s.11.3; ``tolerances.py``); this module computes none.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from v2.research import gates as v2_gates
from v2.research import metrics as v2_metrics

from . import kernel_bridge, representation

GATE_NAMES = (
    "g_integrity",
    "g_failclosed",
    "g_truth_sign",
    "g_monotone",
    "g_validation",
    "g_calibration",
    "g_baseline_valid",
    "g_coef_band",
    "g_threshold_band",
    "g_ba_rel",
)
PARAMETER_SPACE_GATES = ("g_coef_band", "g_threshold_band", "g_truth_sign")
OUTPUT_GATES = ("g_ba_rel", "g_validation", "g_calibration", "g_monotone")
KERNEL_OUTPUT_GATES = ("g_failclosed", "g_monotone", "g_validation", "g_calibration",
                       "g_baseline_valid", "g_ba_rel")
GATES_PURPOSE = "GATES"
INPUT_PROBES = v2_gates.FAILCLOSED_PROBES
TT_PROBE_IDS = tuple(f"TT{i}" for i in range(1, 11))
N_PROBES = len(INPUT_PROBES) + len(TT_PROBE_IDS)
VALID_FEATURES = v2_gates.VALID_FEATURES
THETA_NAMES = ("intercept",) + representation.FEATURE_NAMES
SECONDS = "seconds_since_last_success"


# --------------------------------------------------------------------------------------------
# transform-table probes (s.6.7)
# --------------------------------------------------------------------------------------------
def _dump(artifact: Mapping) -> str:
    # allow_nan: TT4/TT5 must carry the NaN / Infinity tokens a strict loader refuses
    return json.dumps(artifact, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=True) + "\n"


def transform_table_probes(artifact: Mapping) -> list[tuple[str, bytes]]:
    """(probe_id, model bytes) for TT1-TT10, each derived from ``artifact``."""

    def changed(change: Callable[[dict], None]) -> bytes:
        art = copy.deepcopy(dict(artifact))
        change(art["transform_table"])
        return _dump(art).encode("utf-8")

    def set_entry(name: str, key: str, value: float) -> Callable[[dict], None]:
        return lambda table: table[name].__setitem__(key, value)

    probes = [
        ("TT1", changed(set_entry(SECONDS, "scale", 86400.0))),
        ("TT2", changed(set_entry(SECONDS, "scale", 0.0))),
        ("TT3", changed(set_entry(SECONDS, "scale", -900.0))),
        ("TT4", changed(set_entry(SECONDS, "scale", float("nan")))),
        ("TT5", changed(set_entry(SECONDS, "scale", float("inf")))),
        ("TT6", changed(set_entry(SECONDS, "offset", 1.0))),
        ("TT7", changed(set_entry("tls_enabled", "scale", 2.0))),
        ("TT8", changed(lambda table: table.pop(SECONDS))),
        ("TT9", changed(lambda table: table.__setitem__("uptime_seconds", {"offset": 0.0, "scale": 1.0}))),
    ]
    text = _dump(artifact)
    anchor = '"transform_table": {'
    if text.count(anchor) != 1:
        raise ValueError("artifact text has no unique transform_table")
    duplicate = f'"{SECONDS}": {{"offset": 0.0, "scale": 900.0}}, '
    probes.append(("TT10", text.replace(anchor, anchor + duplicate).encode("utf-8")))
    return probes


def run_transform_probes(kernel_module: ModuleType, artifact: Mapping, receipt: Mapping) -> dict:
    """Load each TT probe with a consistent receipt through ``kernel_module.OpsHealthKernel``."""
    from . import freeze  # noqa: PLC0415 -- freeze imports gates' dependencies only

    refused, escaped, messages = 0, [], {}
    tmp = Path(tempfile.mkdtemp(prefix="oac_v3_tt_"))
    try:
        for probe_id, model_bytes in transform_table_probes(artifact):
            directory = tmp / probe_id
            model, rec = freeze.write_pair(directory, model_bytes,
                                           freeze.consistent_receipt(receipt, model_bytes))
            try:
                kernel_module.OpsHealthKernel(model, rec)
            except Exception as exc:  # noqa: BLE001 -- any refusal counts; the type is recorded
                refused += 1
                messages[probe_id] = f"{type(exc).__name__}: {exc}"
            else:
                escaped.append(probe_id)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"refused": refused, "total": len(TT_PROBE_IDS), "escaped": escaped, "messages": messages}


def run_failclosed(advise: Callable[[Any], Any], kernel_module: ModuleType, artifact: Mapping,
                   receipt: Mapping) -> dict:
    inputs = v2_gates.run_failclosed_probes(advise)
    tables = run_transform_probes(kernel_module, artifact, receipt)
    refused = inputs["refused"] + tables["refused"]
    total = inputs["total"] + tables["total"]
    return {"refused": refused, "total": total, "fraction": refused / total,
            "escaped": inputs["escaped"] + tables["escaped"],
            "input_probes": {k: inputs[k] for k in ("refused", "total", "escaped")},
            "transform_table_probes": {k: tables[k] for k in ("refused", "total", "escaped")}}


# --------------------------------------------------------------------------------------------
# scalar gates
# --------------------------------------------------------------------------------------------
g_validation = v2_gates.g_validation
g_calibration = v2_gates.g_calibration
g_baseline_valid = v2_gates.g_baseline_valid
g_monotone = v2_gates.g_monotone


def g_truth_sign(weights: Mapping[str, float]) -> float:
    return representation.sign_agreement(weights)["fraction"]


def g_coef_band(theta: Sequence[float] | None, theta_ref: Sequence[float],
                tolerances: Sequence[float]) -> float:
    if theta is None or len(theta) != 9 or len(theta_ref) != 9 or len(tolerances) != 9:
        return 0.0
    for value, ref, tol in zip(theta, theta_ref, tolerances, strict=True):
        if not (math.isfinite(value) and abs(value - ref) <= tol):
            return 0.0
    return 1.0


def g_threshold_band(k: int | None, k_ref: int, tol_k: int) -> float:
    for value in (k, k_ref, tol_k):
        if isinstance(value, bool) or not isinstance(value, int):
            return 0.0
    return 1.0 if abs(k - k_ref) <= tol_k else 0.0


def g_ba_rel(ba: float | None, ba_ref: float, tol_ba: float) -> float:
    if ba is None:
        return 0.0
    return 1.0 if ba >= ba_ref - tol_ba else 0.0


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


def caught_by(vector: Mapping[str, float], reference: Mapping[str, float]) -> list[str]:
    """Gates whose value is strictly below the reference's own value (s.11.2)."""
    return [name for name in GATE_NAMES if vector[name] < reference[name]]


# --------------------------------------------------------------------------------------------
# fixture evaluation
# --------------------------------------------------------------------------------------------
class GateContext:
    """The gate evaluation rows (the registered ``gates`` split, or DEV rows in tests)."""

    def __init__(self, rows: Sequence[Mapping], name: str):
        self.rows = list(rows)
        self.name = name
        self.labels = [1 if r["label"]["operator_attention_required"] else 0 for r in self.rows]
        self.features = [r["features"] for r in self.rows]


def exact_ba(labels: Sequence[int], decisions: Sequence[int]) -> Fraction | None:
    c = v2_metrics.confusion(labels, decisions)
    pos, neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
    if not pos or not neg:
        return None
    return (Fraction(c["tp"], pos) + Fraction(c["tn"], neg)) / 2


def artifact_parameters(model_bytes: bytes, kernel_module: ModuleType) -> dict:
    """theta on the R scale, threshold index and weights read from the artifact's own bytes."""
    try:
        artifact = kernel_module.loads_strict(model_bytes.decode("utf-8"), kernel_module.ModelArtifactError)
        theta = representation.artifact_theta_r(artifact)
        k = artifact["threshold_index"]
        weights = {name: float(artifact["weights"][name]) for name in representation.FEATURE_NAMES}
    except Exception:  # noqa: BLE001 -- unreadable parameters fail closed
        return {"readable": False, "theta_r": None, "threshold_index": None, "weights": None}
    if isinstance(k, bool) or not isinstance(k, int):
        k = None
    return {"readable": True, "theta_r": theta, "threshold_index": k, "weights": weights}


def outputs(kernel_obj, ctx: GateContext) -> dict:
    evals = [kernel_obj.evaluate(kernel_obj.normalize(f)) for f in ctx.features]
    scores = [e["calibrated_score"] for e in evals]
    decisions = [1 if e["threshold_decision"] else 0 for e in evals]
    return {"scores": scores, "decisions": decisions,
            "advisories": [e["advisory"] for e in evals]}


def evaluate_fixture(model_path: Path | str, receipt_path: Path | str, ctx: GateContext, *,
                     reference: Mapping | None, tolerances: Mapping | None,
                     kernel_module: ModuleType | None = None,
                     receipt_verified: bool | None = None) -> dict:
    """Every s.11.2 gate of one fixture. ``reference`` holds theta_r, threshold_index and
    ba_gates of the frozen v3 (None: the fixture is the reference, every band gate is 1).
    ``receipt_verified`` is precondition (i) of s.11.1 (kernel AND research origin check);
    None means kernel acceptance only (the origin check is the recipe lane's)."""
    module = kernel_module or kernel_bridge.kernel()
    model_path, receipt_path = Path(model_path), Path(receipt_path)
    model_bytes = model_path.read_bytes()
    params = artifact_parameters(model_bytes, module)
    try:
        kernel_obj = module.OpsHealthKernel(model_path, receipt_path)
        integrity, refusal = 1.0, None
    except Exception as exc:  # noqa: BLE001 -- any refusal (or crash) fails integrity
        kernel_obj, integrity, refusal = None, 0.0, f"{type(exc).__name__}: {exc}"
    values = {name: 0.0 for name in GATE_NAMES}
    values["g_integrity"] = integrity
    details: dict[str, Any] = {"integrity_refusal": refusal, "parameters": params}
    if params["readable"]:
        values["g_truth_sign"] = g_truth_sign(params["weights"])
    ba = None
    if kernel_obj is not None:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        fc = run_failclosed(kernel_obj.advise, module, kernel_obj.artifact, receipt)
        mono_value, mono_report = g_monotone(
            lambda f: kernel_obj.advise({"features": f})["operator_attention_score"])
        out = outputs(kernel_obj, ctx)
        ba_exact = exact_ba(ctx.labels, out["decisions"])
        ba = float(ba_exact) if ba_exact is not None else None
        ece = v2_metrics.ece(ctx.labels, out["scores"])
        verified = (integrity == 1.0) if receipt_verified is None else bool(receipt_verified)
        validity = v2_metrics.baseline_validity(ctx.labels, out["decisions"], receipt_verified=verified,
                                                probes_refused=fc["refused"], probes_total=fc["total"])
        values.update({
            "g_failclosed": fc["fraction"],
            "g_monotone": mono_value,
            "g_validation": g_validation(ba),
            "g_calibration": g_calibration(ece),
            "g_baseline_valid": g_baseline_valid(validity),
        })
        details.update({
            "failclosed": {k: fc[k] for k in ("refused", "total", "escaped")},
            "monotone": mono_report,
            "balanced_accuracy": ba,
            "balanced_accuracy_exact": [ba_exact.numerator, ba_exact.denominator] if ba_exact else None,
            "ece": ece,
            "brier": v2_metrics.brier(ctx.labels, out["scores"]),
            "baseline_validity": validity,
            "binary_report": v2_metrics.binary_report(ctx.labels, out["decisions"], validity),
        })
    if reference is None:
        values["g_coef_band"] = 1.0 if params["readable"] else 0.0
        values["g_threshold_band"] = 1.0 if params["threshold_index"] is not None else 0.0
        values["g_ba_rel"] = 1.0 if ba is not None else 0.0
    else:
        if tolerances is None:
            raise ValueError("a non-reference fixture needs the committed tolerance vector")
        values["g_coef_band"] = g_coef_band(params["theta_r"], reference["theta_r"], tolerances["theta"])
        values["g_threshold_band"] = g_threshold_band(params["threshold_index"],
                                                      reference["threshold_index"], tolerances["k"])
        if kernel_obj is not None:
            values["g_ba_rel"] = g_ba_rel(ba, reference["ba_gates"], tolerances["ba"])
    return {"gates": gate_vector(values), "details": details, "context": ctx.name,
            "rows": len(ctx.rows)}


def reference_of(fixture_result: Mapping) -> dict:
    """The reference values that the band gates compare against, from the frozen v3's result."""
    details = fixture_result["details"]
    return {"theta_r": details["parameters"]["theta_r"],
            "threshold_index": details["parameters"]["threshold_index"],
            "ba_gates": details["balanced_accuracy"]}
