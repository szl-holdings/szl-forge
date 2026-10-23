"""v3 mutation suite: the registered catalogue (protocol s.11.4). SYNTHETIC; standard library only.

This module DEFINES and BUILDS the mutants. It does not run them: the passes, the gate
evaluation of each mutant, the kernel-relevant test runs and the BLIND_SPOT statuses are protocol
step 7, after the tolerance vector is committed (s.11.3) and the kernel-relevant test list is
committed (s.11.4, s.13 step 7). ``prepare_run`` holds the refusals that a step-7 run must pass
first; it refuses unless the tolerance vector and its digest are in a commit that precedes the
run (s.11.3), the tree is clean, and the runtime pins hold.

Registered artifact mutants of the frozen v3 (37)
    A-scale0.5-<f>, A-scale1.5-<f>     weight x0.5 and x1.5, per feature            (16)
    A-signflip-<f>                     weight sign flip, per feature                (8)
    A-drop-<f>                         weight set to 0, per feature                 (8)
    A-intercept+1, A-intercept-1       intercept +1 and -1                          (2)
    A-threshold+0.1, A-threshold-0.1   grid index +10 and -10                       (2)
    A-table-seconds-86400              declared seconds scale 900 -> 86400, every weight
                                       unchanged (the kernel must refuse it, and g_coef_band
                                       must catch it, s.7 H2 item 2)            (1)
  Each mutated real number is rounded to 12 decimals (frozen resolution).
Registered receipt mutants (7)
    R-model-hash-flip        model_sha256: the low bit of its first hex digit flipped
    R-wrong-schema           schema -> "szl-oac/ops-health-artifact-receipt/v2"
    R-source-commit-altered  source.commit: the low bit of its last hex digit flipped (no such
                             commit)
    R-metrics-added          a metrics block added (the v2 AM-3 form)
    R-first-parent           source.commit -> the claimed commit C's first parent, whose bound
                             blobs are identical
    R-descendant             source.commit -> a descendant of C whose bound blobs are identical
    R-squash-duplicate       source.commit -> S, a squash-style duplicate of C: same tree and
                             same first parent as C, other commit metadata, built in a scratch
                             clone (never pushed)
Registered kernel-source mutants (8), applied to a temporary copy of v3/ops-health/ops_health.py
(never to the tracked file); each edit target must occur exactly once
    K-no-finiteness, K-no-range, K-no-prohibited-screen, K-inverted-model-hash,
    K-inverted-self-hash, K-no-transform-table-check (the kernel applies whatever table the
    artifact declares), K-no-duplicate-key-refusal, K-authority-true (one value of the
    authority map set to true)
Descriptive mutants (sensitivity profile; not in H2)
    D-scale0.75-<f>, D-scale1.25-<f> (16); D-threshold+0.05, D-threshold-0.05 (2);
    D-recipe-ABL-D, D-recipe-ABL-R, D-recipe-ABL-DR (3; the ablation models with a consistent
    receipt, built from their frozen files at step 7)
Passes (s.11.4, as v2 s.10): pass 1 pairs each artifact mutant with the frozen receipt; pass 2
regenerates the receipt consistently (``freeze.consistent_receipt``), which bypasses integrity.

Order rule (s.11.3: "The mutation code refuses to run unless that digest is in a commit that
precedes it"). The builders accept a DEV artifact or receipt (training split ``dev``; DEV fit
block) freely, so the catalogue can be tested in step 2. For anything else they require the
record that ``prepare_run`` returns (tolerance digest committed, clean tree, runtime pins), so no
mutant of a registered fit can be built before the tolerance vector is committed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import guards, registry, representation

FEATURES = representation.FEATURE_NAMES
WEIGHT_FACTORS = (("scale0.5", 0.5), ("scale1.5", 1.5))
DESCRIPTIVE_FACTORS = (("scale0.75", 0.75), ("scale1.25", 1.25))
INTERCEPT_DELTAS = (("intercept+1", 1.0), ("intercept-1", -1.0))
THRESHOLD_STEPS = (("threshold+0.1", 10), ("threshold-0.1", -10))
DESCRIPTIVE_THRESHOLD_STEPS = (("threshold+0.05", 5), ("threshold-0.05", -5))
TABLE_MUTANT_ID = "A-table-seconds-86400"
WRONG_RECEIPT_SCHEMA = "szl-oac/ops-health-artifact-receipt/v2"
ADDED_METRICS_BLOCK = {
    "validation_balanced_accuracy": 0.99,
    "validation_ece": 0.001,
    "note": "tampered: receipts carry no metrics (v2 AM-3)",
}
RECIPE_SUBSTITUTIONS = ("ABL-D", "ABL-R", "ABL-DR")
TOLERANCES_REL = "v3/results/TOLERANCES.json"
N_ARTIFACT = 37
N_RECEIPT = 7
N_KERNEL = 8

# (id, description, ((old, new), ...)); every `old` must occur exactly once in the kernel source.
KERNEL_MUTANTS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("K-no-finiteness", "input finiteness check removed from _normalize_value",
     (("    if not math.isfinite(number):\n"
       "        raise ModelInputError(f\"{name} must be finite\")\n", ""),)),
    ("K-no-range", "input range check removed from _normalize_value",
     (("    if number < minimum or number > maximum:\n"
       "        raise ModelInputError(f\"{name} must be between {minimum:g} and {maximum:g}\")\n", ""),)),
    ("K-no-prohibited-screen", "prohibited-key screening removed from screen_input (match = None)",
     (("            match = prohibited_match(key)\n", "            match = None\n"),)),
    ("K-inverted-model-hash", "model-hash comparison inverted in OpsHealthKernel (!= becomes ==)",
     (("        if sha256_bytes(model_bytes) != receipt[\"model_sha256\"]:\n",
       "        if sha256_bytes(model_bytes) == receipt[\"model_sha256\"]:\n"),)),
    ("K-inverted-self-hash", "kernel self-hash comparison inverted in OpsHealthKernel (!= becomes ==)",
     (("        if kernel_file_sha256() != receipt[\"kernel_sha256\"]:\n",
       "        if kernel_file_sha256() == receipt[\"kernel_sha256\"]:\n"),)),
    ("K-no-transform-table-check",
     "transform-table check bypassed: the kernel applies whatever table the artifact declares",
     (("    table = validate_transform_table(art[\"transform_table\"])\n",
       "    table = {n: (e[\"offset\"], e[\"scale\"]) for n, e in art[\"transform_table\"].items()}\n"),)),
    ("K-no-duplicate-key-refusal", "duplicate-key refusal removed from the strict JSON hook",
     (("            if key in out:\n"
       "                raise error_cls(f\"duplicate JSON key {_shown_key(key)!r} is not accepted\")\n", ""),)),
    ("K-authority-true", "one value of the authority map set to true (acknowledgement)",
     (("    \"acknowledgement\": False,\n", "    \"acknowledgement\": True,\n"),)),
)


class MutationRefused(RuntimeError):
    pass


DEV_FIT_PREFIX = "dev-fit"


def _is_dev_artifact(artifact: Mapping) -> bool:
    try:
        splits = artifact["training"]["splits"]
        return all(entry["name"] == registry.DEV_SPLIT for entry in splits.values())
    except (KeyError, TypeError, AttributeError):
        return False


def _is_dev_receipt(receipt: Mapping) -> bool:
    try:
        return str(receipt["fit"]["declared_fit"]).startswith(DEV_FIT_PREFIX)
    except (KeyError, TypeError):
        return False


def _require_prepared(prepared: Mapping | None, what: str) -> None:
    ok = (isinstance(prepared, Mapping) and isinstance(prepared.get("tolerances"), Mapping)
          and isinstance(prepared["tolerances"].get("tolerances_sha256"), str)
          and isinstance(prepared.get("runtime_pins"), Mapping)
          and prepared["runtime_pins"].get("pinned") is True)
    if not ok:
        raise MutationRefused(
            f"{what} of a registered fit are built only after prepare_run (tolerance digest "
            "committed, clean tree, runtime pins; prereg s.11.3)")


def q12(value: float) -> float:
    return round(float(value), 12)


# --------------------------------------------------------------------------------------------
# artifact mutants
# --------------------------------------------------------------------------------------------
def _mutant(artifact: Mapping, mutant_id: str, family: str, feature: str | None, field: str,
            before: Any, after: Any, description: str) -> dict:
    mutated = copy.deepcopy(dict(artifact))
    if field == "intercept":
        mutated["intercept"] = after
    elif field == "threshold_index":
        mutated["threshold_index"] = after
    elif field == "transform_table.seconds_since_last_success.scale":
        mutated["transform_table"]["seconds_since_last_success"]["scale"] = after
    else:
        mutated["weights"][feature] = after
    return {"id": mutant_id, "family": family, "feature": feature, "field": field, "before": before,
            "after": after, "description": description, "artifact": mutated}


def _threshold(artifact: Mapping, tag: str, step: int, prefix: str) -> dict:
    index = artifact["threshold_index"]
    new = index + step
    if not 1 <= new <= 99:
        raise MutationRefused(f"{tag} leaves the registered 1..99 grid")
    return _mutant(artifact, f"{prefix}-{tag}", "threshold", None, "threshold_index", index, new,
                   f"grid index {index} -> {new} ({step / 100:+g})")


def artifact_mutants(artifact: Mapping, *, prepared: Mapping | None = None) -> list[dict]:
    """The 37 registered artifact mutants of ``artifact`` (s.11.4)."""
    if not _is_dev_artifact(artifact):
        _require_prepared(prepared, "artifact mutants")
    out = []
    weights = artifact["weights"]
    for tag, factor in WEIGHT_FACTORS:
        for f in FEATURES:
            out.append(_mutant(artifact, f"A-{tag}-{f}", f"weight x{factor:g}", f, f"weights.{f}",
                               weights[f], q12(weights[f] * factor), f"weight of {f} x{factor:g}"))
    for f in FEATURES:
        out.append(_mutant(artifact, f"A-signflip-{f}", "sign flip", f, f"weights.{f}", weights[f],
                           q12(-weights[f]), f"sign of the weight of {f} flipped"))
    for f in FEATURES:
        out.append(_mutant(artifact, f"A-drop-{f}", "drop feature", f, f"weights.{f}", weights[f],
                           0.0, f"feature {f} dropped (weight 0)"))
    for tag, delta in INTERCEPT_DELTAS:
        out.append(_mutant(artifact, f"A-{tag}", "intercept", None, "intercept", artifact["intercept"],
                           q12(artifact["intercept"] + delta), f"intercept {delta:+g}"))
    for tag, step in THRESHOLD_STEPS:
        out.append(_threshold(artifact, tag, step, "A"))
    scale = artifact["transform_table"]["seconds_since_last_success"]["scale"]
    out.append(_mutant(artifact, TABLE_MUTANT_ID, "transform table", None,
                       "transform_table.seconds_since_last_success.scale", scale, 86400.0,
                       "declared seconds scale 900 -> 86400, weights unchanged"))
    ids = [m["id"] for m in out]
    if len(ids) != N_ARTIFACT or len(set(ids)) != N_ARTIFACT:
        raise MutationRefused("the artifact catalogue must hold 37 distinct mutants")
    return out


def descriptive_mutants(artifact: Mapping, *, prepared: Mapping | None = None) -> list[dict]:
    """Descriptive artifact mutants (x0.75, x1.25 per feature; threshold +-0.05): 18."""
    if not _is_dev_artifact(artifact):
        _require_prepared(prepared, "descriptive mutants")
    out = []
    weights = artifact["weights"]
    for tag, factor in DESCRIPTIVE_FACTORS:
        for f in FEATURES:
            out.append(_mutant(artifact, f"D-{tag}-{f}", f"weight x{factor:g}", f, f"weights.{f}",
                               weights[f], q12(weights[f] * factor), f"weight of {f} x{factor:g}"))
    for tag, step in DESCRIPTIVE_THRESHOLD_STEPS:
        out.append(_threshold(artifact, tag, step, "D"))
    return out


def recipe_substitution_mutants(models: Mapping[str, Path]) -> list[dict]:
    """D-recipe-<ABL>: the ablation's frozen model.json presented with a consistent receipt."""
    if set(models) != set(RECIPE_SUBSTITUTIONS):
        raise MutationRefused(f"recipe substitution needs exactly {RECIPE_SUBSTITUTIONS}")
    return [{"id": f"D-recipe-{name}", "family": "recipe substitution", "model_path": str(models[name])}
            for name in RECIPE_SUBSTITUTIONS]


# --------------------------------------------------------------------------------------------
# receipt mutants
# --------------------------------------------------------------------------------------------
def _flip_low_bit_hex(digit: str) -> str:
    return format(int(digit, 16) ^ 1, "x")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def first_parent(repo: Path, commit: str) -> str:
    return _git(repo, "rev-parse", f"{commit}^1")


def bound_blobs_identical(repo: Path, a: str, b: str, paths: Sequence[str]) -> bool:
    """True iff every bound path has the same blob id at commits a and b."""
    for rel in paths:
        if _git(repo, "rev-parse", f"{a}:{rel}") != _git(repo, "rev-parse", f"{b}:{rel}"):
            return False
    return True


def build_squash_duplicate(repo: Path, commit: str, scratch: Path) -> dict:
    """S: same tree and same first parent as ``commit``, other metadata, in a scratch clone."""
    scratch = guards.refuse_sealed_path(scratch)
    if Path(scratch).exists():
        raise MutationRefused("the scratch clone path must not exist")
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", "--no-hardlinks", str(repo), str(scratch)],
                   check=True, capture_output=True, text=True)
    tree = _git(scratch, "rev-parse", f"{commit}^{{tree}}")
    parent = _git(scratch, "rev-parse", f"{commit}^1")
    env_cmd = ["git", "-C", str(scratch), "-c", "user.name=oac-v3-mutation",
               "-c", "user.email=oac-v3-mutation@localhost", "-c", "commit.gpgsign=false",
               "commit-tree", tree, "-p", parent, "-m",
               f"squash-style duplicate of {commit} (mutation suite; never pushed)"]
    duplicate = subprocess.run(env_cmd, check=True, capture_output=True, text=True).stdout.strip()
    if duplicate == commit:
        raise MutationRefused("the duplicate must have another commit id")
    return {"commit": duplicate, "tree": tree, "first_parent": parent, "scratch": str(scratch)}


def receipt_mutants(receipt: Mapping, *, first_parent_commit: str, descendant_commit: str,
                    squash_duplicate_commit: str, prepared: Mapping | None = None) -> list[dict]:
    """The 7 registered receipt mutants of ``receipt`` (the three git ids are supplied)."""
    if not _is_dev_receipt(receipt):
        _require_prepared(prepared, "receipt mutants")
    out = []

    def add(mutant_id: str, description: str, change) -> None:
        mutated = json.loads(json.dumps(receipt))
        change(mutated)
        out.append({"id": mutant_id, "description": description, "receipt": mutated})

    model_sha = receipt["model_sha256"]
    flipped = _flip_low_bit_hex(model_sha[0]) + model_sha[1:]
    add("R-model-hash-flip", "model_sha256 first hex digit low bit flipped",
        lambda r: r.update(model_sha256=flipped))
    add("R-wrong-schema", f"schema -> {WRONG_RECEIPT_SCHEMA!r}",
        lambda r: r.update(schema=WRONG_RECEIPT_SCHEMA))
    commit = receipt["source"]["commit"]
    altered = commit[:-1] + _flip_low_bit_hex(commit[-1])
    add("R-source-commit-altered", "source.commit last hex digit low bit flipped",
        lambda r: r["source"].update(commit=altered))
    add("R-metrics-added", "a metrics block added (v2 AM-3 form)",
        lambda r: r.update(metrics=dict(ADDED_METRICS_BLOCK)))
    for mutant_id, target, text in (
        ("R-first-parent", first_parent_commit, "the claimed commit's first parent"),
        ("R-descendant", descendant_commit, "a descendant with identical bound blobs"),
        ("R-squash-duplicate", squash_duplicate_commit, "a squash-style duplicate (same tree and parent)"),
    ):
        if not isinstance(target, str) or len(target) != 40 or target == commit:
            raise MutationRefused(f"{mutant_id} needs a 40-hex commit other than the claimed one")
        add(mutant_id, f"source.commit -> {text}", lambda r, t=target: r["source"].update(commit=t))
    if len(out) != N_RECEIPT:
        raise MutationRefused("the receipt catalogue must hold 7 mutants")
    return out


# --------------------------------------------------------------------------------------------
# kernel-source mutants
# --------------------------------------------------------------------------------------------
def apply_kernel_edits(text: str, edits: Sequence[tuple[str, str]]) -> str:
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise MutationRefused(f"kernel mutation target occurs {count} times: {old[:60]!r}")
        text = text.replace(old, new)
    return text


def kernel_mutant_sources(text: str) -> list[dict]:
    out = []
    for mutant_id, description, edits in KERNEL_MUTANTS:
        out.append({"id": mutant_id, "description": description, "source": apply_kernel_edits(text, edits)})
    if len(out) != N_KERNEL:
        raise MutationRefused("the kernel catalogue must hold 8 mutants")
    return out


def write_kernel_copy(directory: Path, name: str, text: str) -> Path:
    directory = guards.refuse_sealed_path(directory)
    sub = Path(directory) / name
    sub.mkdir(parents=True, exist_ok=False)
    path = sub / "ops_health.py"
    path.write_bytes(text.encode("utf-8"))
    if path.resolve() == registry.KERNEL_PATH.resolve():
        raise MutationRefused("kernel mutants are never written over the tracked kernel")
    return path


def catalogue_summary(artifact: Mapping, kernel_text: str) -> dict:
    """Ids and families of every registered and descriptive mutant (no gate is evaluated)."""
    arts = artifact_mutants(artifact)
    desc = descriptive_mutants(artifact)
    kern = kernel_mutant_sources(kernel_text)
    return {
        "artifact": [{"id": m["id"], "family": m["family"]} for m in arts],
        "receipt": ["R-model-hash-flip", "R-wrong-schema", "R-source-commit-altered", "R-metrics-added",
                    "R-first-parent", "R-descendant", "R-squash-duplicate"],
        "kernel": [{"id": m["id"], "source_sha256": hashlib.sha256(m["source"].encode()).hexdigest()}
                   for m in kern],
        "descriptive": [m["id"] for m in desc] + [f"D-recipe-{n}" for n in RECIPE_SUBSTITUTIONS],
        "counts": {"artifact": len(arts), "receipt": N_RECEIPT, "kernel": len(kern),
                   "descriptive": len(desc) + len(RECIPE_SUBSTITUTIONS)},
    }


# --------------------------------------------------------------------------------------------
# refusals that precede any step-7 run
# --------------------------------------------------------------------------------------------
def require_tolerances_committed(root: Path = registry.ROOT, rel: str = TOLERANCES_REL) -> dict:
    """Refuse unless the tolerance vector is tracked at HEAD, unchanged in the working tree, and
    its recorded sha256 equals the digest of its canonical content (s.11.3)."""
    from . import tolerances  # noqa: PLC0415

    try:
        prefix = _git(root, "rev-parse", "--show-prefix")
        head = _git(root, "rev-parse", "HEAD")
        blob = subprocess.run(["git", "-C", str(root), "cat-file", "blob", f"HEAD:{prefix}{rel}"],
                              check=True, capture_output=True).stdout
        dirty = _git(root, "status", "--porcelain", "--", rel)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise MutationRefused("the tolerance vector is not in a commit that precedes this run") from exc
    if dirty:
        raise MutationRefused("the tolerance vector differs from its commit; refusing")
    try:
        record = json.loads(blob.decode("utf-8"))
        vector, recorded = record["vector"], record["sha256"]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise MutationRefused("the committed tolerance record is malformed") from exc
    if tolerances.digest(vector) != recorded:
        raise MutationRefused("the committed tolerance digest does not match its vector")
    commits = _git(root, "log", "--format=%H", "--", rel).splitlines()
    return {"head": head, "tolerances_sha256": recorded, "introduced_by": commits[-1] if commits else None}


def prepare_run(root: Path = registry.ROOT) -> dict:
    """Every refusal of a step-7 mutation run, in order. Returns what it checked."""
    tol = require_tolerances_committed(root)
    dirty = _git(root, "status", "--porcelain", "--", "v3")
    if dirty:
        raise MutationRefused("the mutation suite runs from a clean tracked tree only")
    pins = guards.require_runtime_pins()
    return {"tolerances": tol, "runtime_pins": pins}
