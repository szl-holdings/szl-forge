"""Score calibrators: none, Platt and isotonic (PREREGISTRATION section 6).  Standard library only.

Platt
    Two-parameter logistic regression of the label on u = logit(clip(score, 1e-15, 1 - 1e-15)),
    fit by the same Newton/IRLS routine as the model (lr.fit, l2 = 0, max|delta| < 1e-10 or
    100 iterations).  Calibrated score = sigmoid(a * u + b).  Plain 0/1 targets.
Isotonic
    Pool-adjacent-violators on the raw score quantized to 12 decimals (the frozen resolution).
    Ties are pooled first: rows with equal quantized score form one initial block.  Blocks are
    then merged while the previous block's mean label is >= the current block's mean (exact
    integer cross-multiplication, so the result is exact and deterministic); the fitted block
    values are therefore strictly increasing.  Step prediction (no interpolation): the value of
    the last block whose minimum score is <= s; the first block's value if s is below all blocks.
Frozen resolution
    Every fitted parameter is rounded to 12 decimals as soon as it is fitted (the resolution of
    model.json), and all downstream steps use the rounded values.  Application is delegated to
    the kernel's ``apply_calibrator`` so research and kernel arithmetic are identical.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import kernel_bridge, lr

KINDS = ("none", "platt", "isotonic")
KIND_ORDER = {kind: index for index, kind in enumerate(KINDS)}  # registered tie-break order


class CalibrationError(ValueError):
    pass


def _k():
    return kernel_bridge.kernel()


def identity() -> dict:
    return {"kind": "none", "params": {}}


def fit_platt(scores: Sequence[float], labels: Sequence[int]) -> tuple[dict, dict]:
    """Returns (calibrator, diagnostics)."""
    k = _k()
    if len(scores) != len(labels) or not scores:
        raise CalibrationError("scores and labels must be non-empty and aligned")
    if len(set(int(v) for v in labels)) < 2:
        raise CalibrationError("Platt scaling needs both classes")
    u = [k.logit_clipped(float(s)) for s in scores]
    result = lr.fit([u], labels, 0.0)
    a = round(result["weights"][0], 12)
    b = round(result["intercept"], 12)
    if abs(a) > k.PARAM_LIMIT or abs(b) > k.PARAM_LIMIT:
        raise CalibrationError("Platt parameters exceed the artifact bound")
    diagnostics = {
        key: result[key]
        for key in ("iterations", "converged", "stop_reason", "gradient_max_abs", "rows")
    }
    return {"kind": "platt", "params": {"a": a, "b": b}}, diagnostics


def fit_isotonic(scores: Sequence[float], labels: Sequence[int]) -> tuple[dict, dict]:
    """Returns (calibrator, diagnostics).  Exact PAV with ties pooled first."""
    k = _k()
    if len(scores) != len(labels) or not scores:
        raise CalibrationError("scores and labels must be non-empty and aligned")
    keys = [k.isotonic_key(float(s)) for s in scores]
    ys = [int(v) for v in labels]
    order = sorted(range(len(keys)), key=keys.__getitem__)
    initial: list[list] = []  # [min_key, positives, count]
    for i in order:
        if initial and initial[-1][0] == keys[i]:
            initial[-1][1] += ys[i]
            initial[-1][2] += 1
        else:
            initial.append([keys[i], ys[i], 1])
    stack: list[list] = []
    for block in initial:
        stack.append(list(block))
        while len(stack) >= 2 and stack[-2][1] * stack[-1][2] >= stack[-1][1] * stack[-2][2]:
            top = stack.pop()
            stack[-1][1] += top[1]
            stack[-1][2] += top[2]
    if len(stack) > k.MAX_ISOTONIC_BLOCKS:
        raise CalibrationError("isotonic fit exceeds the artifact block bound")
    calibrator = {
        "kind": "isotonic",
        "params": {
            "thresholds": [block[0] for block in stack],
            "values": [round(block[1] / block[2], 12) for block in stack],
        },
    }
    diagnostics = {
        "rows": len(keys),
        "tie_blocks": len(initial),
        "blocks": len(stack),
        "block_counts": [block[2] for block in stack],
    }
    return calibrator, diagnostics


def fit(kind: str, scores: Sequence[float], labels: Sequence[int]) -> tuple[dict, dict]:
    if kind == "none":
        return identity(), {"rows": len(scores)}
    if kind == "platt":
        return fit_platt(scores, labels)
    if kind == "isotonic":
        return fit_isotonic(scores, labels)
    raise CalibrationError(f"unknown calibrator kind {kind!r}")


def apply(calibrator: Mapping, score: float) -> float:
    return _k().apply_calibrator(calibrator, score)


def apply_many(calibrator: Mapping, scores: Sequence[float]) -> list[float]:
    fn = _k().apply_calibrator
    return [fn(calibrator, s) for s in scores]


def serialize(calibrator: Mapping) -> dict:
    """Artifact form {kind, params}, validated by the kernel."""
    return _k().validate_calibrator(calibrator)


def deserialize(obj: Mapping) -> dict:
    return _k().validate_calibrator(obj)
