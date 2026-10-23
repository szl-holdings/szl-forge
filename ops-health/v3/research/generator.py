"""ops-health-generator/3.0.0: synthetic operational transport telemetry (SYNTHETIC only).

Implements ``v3/PREREGISTRATION.md`` sections 2, 3.1 and 3.2 (as amended by
``v3/REREGISTRATION_r1.md``: the beacon round is R' = 6659606). Standard library only.

What is reused (s.3.1), unchanged from v2's ``ops-health-generator/2.0.0``
(``v2/research/generator.py``, sha256 a502d364...4384): the per-row draws, their order, the
rounding, the four shift regimes, the label rule, the row format and the canonical JSONL writer.
The functions ``truth_linear``, ``truth_probability``, ``_weighted_failure_count`` and
``draw_row`` are copied from it; the parity tests of s.3.3 compare them with v2's code and v2's
recorded digests. The rows are operational counters only.

Per-row draw order of a regime (identical to v2; one ``random.Random`` per stream)
    1. listener_running           int(rng.random() < 0.94)
    2. tls_enabled                int(rng.random() < 0.90)     (0.30 under missing_tls)
    3. peer_allowlist_configured  int(rng.random() < 0.93)
    4. queue_utilization          round(min(1.0, rng.betavariate(1.25, 4.5)), 6)
                                  (betavariate(4.0, 1.5) under queue_saturation)
    5. consecutive_failures       v1 weighted draw (failure_bursts: b = rng.random(); b < 0.35 ->
                                  rng.randint(3, 20), else the weighted draw)
    6. seconds_since_last_success round(min(86400.0, rng.expovariate(1/900)), 6)
    7. ledger_integrity_ok        int(rng.random() < 0.985)
    8. configuration_valid        int(rng.random() < 0.96)
    (clock_skew: 8b. skew = rng.uniform(0.0, 7200.0); observed seconds =
     round(min(86400.0, latent + skew), 6))
    9. label                      rng.random() < p(latent features)

What is added (s.3.2), none of it changes the truth
    * Namespace ``oac-ops-health-v3`` and master seed M3 = 20261013. A non-sealed split s uses
      ``random.Random(int.from_bytes(sha256(f"oac-ops-health-v3:{M3}:{s}".encode()).digest()[:8],
      "big"))``.
    * Training-design mode D (training splits only; nominal regime). Per row, in this order:
      steps 1-5 as nominal; ``s_nom = round(min(86400, rng.expovariate(1/900)), 6)``;
      ``b = rng.random()``; ``s_wide = round(rng.uniform(0.0, 86400.0), 6)``; steps 7-8 as
      nominal; ``u = rng.random()``. The D arm's latent seconds are ``s_wide`` if b < 0.25, else
      ``s_nom``; the no-D arm's are always ``s_nom``. Each arm's label is ``u < p(that arm's
      latent features)``; observed equals latent. Both arms come from one stream, so they differ
      only in the rows with b < 0.25. ``U(0, 86400)`` is read as ``rng.uniform(0.0, 86400.0)``,
      the same form v2 uses for ``U(0, 7200)``.
    * Clock-skew twin mode (the two sealed H3 splits only). The v2 clock_skew draw, unchanged; the
      observed row goes to ``h3_clock_skew`` and the same row with its latent seconds to
      ``h3_clock_skew_latent`` (one stream, the one of ``h3_clock_skew``). Both carry the label
      from the latent features. The skew itself is never written.
    * Beacon-salted seeds for sealed splits: seed material
      ``f"oac-ops-health-v3:{M3}:{s}:{SALT}"``, where SALT is the randomness of round R' of the
      registered drand chain as 64 lower-case hex characters. ``generate_sealed`` accepts SALT
      only as a ``VerifiedRound`` produced by the frozen beacon verifier (``v3.research.beacon``,
      built by its own lane; until it exists, sealed generation is impossible) for the
      registered chain and round, and refuses everything else. No SALT exists before R''s
      scheduled time (2026-11-21T00:00:00Z), so no sealed row can be generated before then.
    * The v2 seal guard (``guards.check_namespace``): namespace ``oac-ops-health-v2`` with a v2
      sealed split name is refused, and every data path refuses ``v2/data/sealed/``.

Row format: v2's, unchanged (schema ``szl-oac/ops-health-observation/v2``, keys schema,
synthetic, sample_id, regime, features, label). The observation format and the eight fields are
not changed by v3 (s.3.1, s.6.7 item 2); the generator version and the split table identify v3
data. ``regime`` names the v2 regime of the draws; the design (D, no-D, twin) is a property of
the split and is recorded in the split table and the data manifest.

DEV rule (s.15.5): DEV rows are generated only under the split name ``dev``, at most 2,048 rows
per regime per DEV run (``guards.DEV_LEDGER``). The v2 parity mode returns digests only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from . import guards, registry

GENERATOR_VERSION = "ops-health-generator/3.0.0"
SEED_NAMESPACE = registry.SEED_NAMESPACE
MASTER_SEED = registry.MASTER_SEED
ROW_SCHEMA = "szl-oac/ops-health-observation/v2"
V1_ROW_SCHEMA = "szl-oac/transport-health-observation/v1"
V1_SEED = 2500
V1_SPLIT_ROWS = (("train", 768), ("validation", 192), ("test", 240))
REGIMES = ("nominal", "queue_saturation", "failure_bursts", "clock_skew", "missing_tls")
MODES = ("plain", "D", "noD", "twin_observed", "twin_latent")
PI_D = registry.PI_D
SECONDS_MAX = 86400.0

# (name, minimum, maximum, kind) -- identical to the v1 and v2 FEATURE_SPECS.
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
_HEX64 = re.compile(r"[0-9a-f]{64}")


class GeneratorRefused(guards.GuardError):
    """A generation request outside the registered modes, splits or seeds."""


# ------------------------------------------------------------------------------------------
# generator truth (s.2): copied from v2, same expression and evaluation order as v1
# ------------------------------------------------------------------------------------------
def normalize(features: Mapping[str, Any]) -> list[float]:
    """v1 arithmetic: binary -> float(value); continuous -> (x - min) / (max - min)."""
    out = []
    for name, minimum, maximum, kind in FEATURE_SPECS:
        value = features[name]
        if kind == "binary":
            out.append(1.0 if value else 0.0)
        else:
            out.append((float(value) - minimum) / (maximum - minimum))
    return out


def truth_linear(features: Mapping[str, Any]) -> float:
    """Generator-truth logit z(x), same expression and evaluation order as v1 and v2."""
    values = dict(zip(FEATURE_NAMES, normalize(features), strict=True))
    return (
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


def truth_probability(features: Mapping[str, Any]) -> float:
    return 1.0 / (1.0 + math.exp(-truth_linear(features)))


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


def draw_row(rng: random.Random, regime: str) -> tuple[dict, dict, bool]:
    """One v2 row (copied unchanged). Returns (observed_features, latent_features, label)."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    latent: dict[str, int | float] = {}
    latent["listener_running"] = int(rng.random() < 0.94)
    latent["tls_enabled"] = int(rng.random() < (0.30 if regime == "missing_tls" else 0.90))
    latent["peer_allowlist_configured"] = int(rng.random() < 0.93)
    if regime == "queue_saturation":
        latent["queue_utilization"] = round(min(1.0, rng.betavariate(4.0, 1.5)), 6)
    else:
        latent["queue_utilization"] = round(min(1.0, rng.betavariate(1.25, 4.5)), 6)
    if regime == "failure_bursts":
        if rng.random() < 0.35:
            latent["consecutive_failures"] = rng.randint(3, 20)
        else:
            latent["consecutive_failures"] = _weighted_failure_count(rng)
    else:
        latent["consecutive_failures"] = _weighted_failure_count(rng)
    latent["seconds_since_last_success"] = round(min(86400.0, rng.expovariate(1.0 / 900.0)), 6)
    latent["ledger_integrity_ok"] = int(rng.random() < 0.985)
    latent["configuration_valid"] = int(rng.random() < 0.96)
    observed = dict(latent)
    if regime == "clock_skew":
        skew = rng.uniform(0.0, 7200.0)
        observed["seconds_since_last_success"] = round(
            min(86400.0, latent["seconds_since_last_success"] + skew), 6
        )
    label = rng.random() < truth_probability(latent)
    return observed, latent, label


def draw_row_d(rng: random.Random) -> dict:
    """One training-design-D draw (s.3.2 item 3; nominal regime). Observed equals latent.

    Returns {"D": (features, label), "noD": (features, label), "b": b, "s_nom": s_nom,
    "s_wide": s_wide}.
    """
    base: dict[str, int | float] = {}
    base["listener_running"] = int(rng.random() < 0.94)
    base["tls_enabled"] = int(rng.random() < 0.90)
    base["peer_allowlist_configured"] = int(rng.random() < 0.93)
    base["queue_utilization"] = round(min(1.0, rng.betavariate(1.25, 4.5)), 6)
    base["consecutive_failures"] = _weighted_failure_count(rng)
    s_nom = round(min(SECONDS_MAX, rng.expovariate(1.0 / 900.0)), 6)
    b = rng.random()
    s_wide = round(rng.uniform(0.0, SECONDS_MAX), 6)
    ledger_ok = int(rng.random() < 0.985)
    config_ok = int(rng.random() < 0.96)
    u = rng.random()
    arms = {}
    for arm, seconds in (("D", s_wide if b < PI_D else s_nom), ("noD", s_nom)):
        features = dict(base)
        features["seconds_since_last_success"] = seconds
        features["ledger_integrity_ok"] = ledger_ok
        features["configuration_valid"] = config_ok
        # key order equal to FEATURE_NAMES (the JSONL writer sorts keys anyway)
        features = {name: features[name] for name in FEATURE_NAMES}
        arms[arm] = (features, u < truth_probability(features))
    return {"D": arms["D"], "noD": arms["noD"], "b": b, "s_nom": s_nom, "s_wide": s_wide}


def make_row(split: str, index: int, regime: str, features: dict, label: bool) -> dict:
    """v2's row format, unchanged."""
    return {
        "schema": ROW_SCHEMA,
        "synthetic": True,
        "sample_id": f"{split}-{index:06d}",
        "regime": regime,
        "features": features,
        "label": {"operator_attention_required": bool(label)},
    }


# ------------------------------------------------------------------------------------------
# seeds (s.3.2 items 1 and 2)
# ------------------------------------------------------------------------------------------
def derive_seed(split: str, *, namespace: str = SEED_NAMESPACE, master_seed: int = MASTER_SEED) -> int:
    """Non-sealed seed of split ``split``. Refuses sealed splits and the v2 sealed names."""
    namespace = guards.check_namespace(namespace, split)
    if namespace == SEED_NAMESPACE:
        if master_seed != MASTER_SEED:
            raise GeneratorRefused("the v3 namespace is used only with M3 = 20261013")
        if guards.fold_name(split) in {guards.fold_name(s) for s in registry.SEALED_SPLITS}:
            raise GeneratorRefused(
                "a v3 sealed split has no unsalted seed; its seed material needs SALT (prereg s.3.2)"
            )
    elif master_seed != registry.V2_MASTER_SEED:
        raise GeneratorRefused("the v2 namespace is used only with M = 20260926 (parity, s.3.3)")
    material = f"{namespace}:{master_seed}:{split}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def sealed_seed_material(split: str, salt: str) -> str:
    """The registered sealed seed material template (s.3.2 item 2). No rows are drawn here."""
    if split not in registry.SEALED_SPLITS:
        raise GeneratorRefused("not a v3 sealed split")
    if not isinstance(salt, str) or not _HEX64.fullmatch(salt):
        raise GeneratorRefused("SALT must be 64 lower-case hex characters")
    return f"{SEED_NAMESPACE}:{MASTER_SEED}:{split}:{salt}"


def sealed_seed(split: str, salt: str) -> int:
    material = sealed_seed_material(split, salt).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


# ------------------------------------------------------------------------------------------
# streams
# ------------------------------------------------------------------------------------------
def _stream(seed: int, split_name: str, n_rows: int, regime: str, mode: str) -> Iterator[dict]:
    """Rows 0..n_rows-1 of one file (``split_name`` is written into sample_id)."""
    if mode not in MODES:
        raise GeneratorRefused(f"unknown mode {mode!r}")
    if mode in ("D", "noD") and regime != "nominal":
        raise GeneratorRefused("training design D exists only in the nominal regime (s.3.2 item 3)")
    if mode.startswith("twin") and regime != "clock_skew":
        raise GeneratorRefused("twin mode exists only for the v2 clock_skew draw (s.3.2 item 4)")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 0:
        raise GeneratorRefused("n_rows must be a non-negative integer")
    rng = random.Random(seed)
    for index in range(n_rows):
        if mode in ("D", "noD"):
            features, label = draw_row_d(rng)[mode]
        else:
            observed, latent, label = draw_row(rng, regime)
            features = latent if mode == "twin_latent" else observed
        yield make_row(split_name, index, regime, features, label)


def _paired_stream(seed: int, names: tuple[str, str], n_rows: int, regime: str,
                   modes: tuple[str, str]) -> tuple[list[dict], list[dict]]:
    """Both files of a paired split from ONE stream (D/noD, or twin observed/latent)."""
    rng = random.Random(seed)
    first, second = [], []
    for index in range(n_rows):
        if modes == ("D", "noD"):
            draw = draw_row_d(rng)
            (fa, la), (fb, lb) = draw["D"], draw["noD"]
        elif modes == ("twin_observed", "twin_latent"):
            observed, latent, label = draw_row(rng, regime)
            fa, la, fb, lb = observed, label, latent, label
        else:
            raise GeneratorRefused("unknown paired modes")
        first.append(make_row(names[0], index, regime, fa, la))
        second.append(make_row(names[1], index, regime, fb, lb))
    return first, second


# ------------------------------------------------------------------------------------------
# registered non-sealed splits (s.4)
# ------------------------------------------------------------------------------------------
def split_definition(name: str) -> dict:
    if name not in registry.SPLIT_TABLE:
        raise GeneratorRefused(f"unknown split {name!r}")
    regime, mode, rows, sealed, stream, use = registry.SPLIT_TABLE[name]
    return {"name": name, "regime": regime, "mode": mode, "rows": rows, "sealed": sealed,
            "stream": stream, "use": use}


def generate_split(name: str) -> list[dict]:
    """Rows of a registered NON-SEALED split (s.4). Sealed splits are refused here."""
    d = split_definition(name)
    if d["sealed"]:
        raise GeneratorRefused("sealed splits are generated only by generate_sealed (s.13 step 9)")
    seed = derive_seed(d["stream"])
    return list(_stream(seed, name, d["rows"], d["regime"], d["mode"]))


def split_bytes(name: str) -> bytes:
    return jsonl_bytes(generate_split(name))


def split_sha256(name: str) -> str:
    """sha256 of a non-sealed split's canonical JSONL bytes, streamed (no rows are kept)."""
    d = split_definition(name)
    if d["sealed"]:
        raise GeneratorRefused("sealed splits are generated only by generate_sealed (s.13 step 9)")
    seed = derive_seed(d["stream"])
    digest = hashlib.sha256()
    for row in _stream(seed, name, d["rows"], d["regime"], d["mode"]):
        digest.update(_row_line(row))
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------
# sealed splits (s.3.2 item 2, s.3.4, s.13 step 9): written now, runnable only after R'
# ------------------------------------------------------------------------------------------
def _verified_salt(verified_round: Any) -> str:
    """SALT from the frozen beacon verifier's output, or a refusal (s.3.3 last beacon item)."""
    try:
        from . import beacon  # noqa: PLC0415 -- the frozen verifier (its own lane)
    except ImportError as exc:
        raise GeneratorRefused(
            "sealed generation needs the frozen beacon verifier (v3.research.beacon); it is absent"
        ) from exc
    verified_type = getattr(beacon, "VerifiedRound", None)
    if not isinstance(verified_type, type) or type(verified_round) is not verified_type:
        raise GeneratorRefused("SALT is accepted only as the beacon verifier's VerifiedRound output")
    fields = {}
    for key in ("chain_hash", "round", "randomness", "signature", "verifier_sha256"):
        fields[key] = getattr(verified_round, key, None)
    if fields["chain_hash"] != registry.CHAIN_HASH:
        raise GeneratorRefused("SALT is accepted only for the registered chain")
    if isinstance(fields["round"], bool) or fields["round"] != registry.BEACON_ROUND:
        raise GeneratorRefused("SALT is accepted only for the registered round R' = 6659606")
    salt, signature = fields["randomness"], fields["signature"]
    if not isinstance(salt, str) or not _HEX64.fullmatch(salt):
        raise GeneratorRefused("SALT must be 64 lower-case hex characters")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]+", signature) or len(signature) % 2:
        raise GeneratorRefused("the verified round carries no hex signature")
    if hashlib.sha256(bytes.fromhex(signature)).hexdigest() != salt:
        raise GeneratorRefused("randomness is not sha256(signature)")
    source_sha = getattr(beacon, "source_sha256", None)
    if not callable(source_sha) or fields["verifier_sha256"] != source_sha():
        raise GeneratorRefused("the verified round was not produced by this tree's beacon verifier")
    return salt


def generate_sealed(name: str, verified_round: Any) -> list[dict] | tuple[list[dict], list[dict]]:
    """Rows of a v3 sealed split, only from a verified round R' (opening run only).

    ``h3_clock_skew`` returns both twin files (observed, latent) from its one stream.
    """
    d = split_definition(name)
    if not d["sealed"]:
        raise GeneratorRefused("not a sealed split")
    if name == "h3_clock_skew_latent":
        raise GeneratorRefused("the latent twin is written with h3_clock_skew from its stream")
    salt = _verified_salt(verified_round)
    seed = sealed_seed(d["stream"], salt)
    if name == "h3_clock_skew":
        return _paired_stream(seed, ("h3_clock_skew", "h3_clock_skew_latent"), d["rows"],
                              "clock_skew", ("twin_observed", "twin_latent"))
    return list(_stream(seed, name, d["rows"], d["regime"], d["mode"]))


# ------------------------------------------------------------------------------------------
# DEV data (s.15.5)
# ------------------------------------------------------------------------------------------
def _dev_record(namespace: str, master_seed: int, regime: str, mode: str, draws: int,
                rows_per_draw: int) -> None:
    guards.DEV_LEDGER.record(namespace=namespace, master_seed=master_seed, regime=regime,
                             mode=mode, draws=draws, rows_per_draw=rows_per_draw)


def _default_master_seed(namespace: str, master_seed: int | None) -> int:
    if master_seed is not None:
        return master_seed
    return MASTER_SEED if namespace == SEED_NAMESPACE else registry.V2_MASTER_SEED


def dev_rows(n_rows: int, regime: str = "nominal", *, namespace: str = SEED_NAMESPACE,
             master_seed: int | None = None) -> list[dict]:
    """DEV rows of a plain regime: split name 'dev', within the DEV cap."""
    master_seed = _default_master_seed(namespace, master_seed)
    seed = derive_seed(registry.DEV_SPLIT, namespace=namespace, master_seed=master_seed)
    _dev_record(namespace, master_seed, regime, "plain", n_rows, 1)
    return list(_stream(seed, registry.DEV_SPLIT, n_rows, regime, "plain"))


def dev_rows_d(n_rows: int) -> tuple[list[dict], list[dict]]:
    """DEV rows of design D: (D arm, no-D arm) from one stream; both count against the cap."""
    seed = derive_seed(registry.DEV_SPLIT)
    _dev_record(SEED_NAMESPACE, MASTER_SEED, "nominal", "D+noD", n_rows, 2)
    return _paired_stream(seed, (registry.DEV_SPLIT, registry.DEV_SPLIT), n_rows, "nominal",
                          ("D", "noD"))


def dev_draws_d(n_rows: int) -> list[dict]:
    """The raw D-mode draws behind ``dev_rows_d`` (b, s_nom, s_wide), for the D-mode check."""
    seed = derive_seed(registry.DEV_SPLIT)
    _dev_record(SEED_NAMESPACE, MASTER_SEED, "nominal", "D+noD", n_rows, 2)
    rng = random.Random(seed)
    return [draw_row_d(rng) for _ in range(n_rows)]


def dev_rows_twin(n_rows: int, *, namespace: str = SEED_NAMESPACE,
                  master_seed: int | None = None) -> tuple[list[dict], list[dict]]:
    """DEV rows of twin mode: (observed, latent) from one clock_skew stream."""
    master_seed = _default_master_seed(namespace, master_seed)
    seed = derive_seed(registry.DEV_SPLIT, namespace=namespace, master_seed=master_seed)
    _dev_record(namespace, master_seed, "clock_skew", "twin", n_rows, 2)
    return _paired_stream(seed, (registry.DEV_SPLIT, registry.DEV_SPLIT), n_rows, "clock_skew",
                          ("twin_observed", "twin_latent"))


# ------------------------------------------------------------------------------------------
# parity modes (s.3.3)
# ------------------------------------------------------------------------------------------
def v2_parity_digest(split: str) -> dict:
    """v2 parity, non-sealed only: sha256 and size of the v2 split regenerated with namespace
    oac-ops-health-v2 and M = 20260926. Rows are hashed as they are drawn and never returned."""
    if split not in registry.V2_PARITY_SPLITS:
        raise GeneratorRefused("v2 parity covers train, calibration, conformal and validation only")
    sizes = {"train": 16384, "calibration": 4096, "conformal": 4096, "validation": 4096}
    seed = derive_seed(split, namespace=registry.V2_SEED_NAMESPACE, master_seed=registry.V2_MASTER_SEED)
    digest = hashlib.sha256()
    size = 0
    for row in _stream(seed, split, sizes[split], "nominal", "plain"):
        line = _row_line(row)
        digest.update(line)
        size += len(line)
    return {"split": split, "rows": sizes[split], "sha256": digest.hexdigest(), "bytes": size}


def generate_v1_parity() -> dict[str, list[dict]]:
    """v1-parity mode (as v2): seed 2500, one sequential Random, v1 row schema, no regime field."""
    rng = random.Random(V1_SEED)
    splits: dict[str, list[dict]] = {}
    for split, row_count in V1_SPLIT_ROWS:
        rows = []
        for index in range(row_count):
            observed, _latent, label = draw_row(rng, "nominal")
            rows.append(
                {
                    "schema": V1_ROW_SCHEMA,
                    "synthetic": True,
                    "sample_id": f"{split}-{index:06d}",
                    "features": observed,
                    "label": {"operator_attention_required": label},
                }
            )
        splits[split] = rows
    return splits


# ------------------------------------------------------------------------------------------
# canonical JSONL (v1's and v2's writer)
# ------------------------------------------------------------------------------------------
def _row_line(row: Mapping[str, Any]) -> bytes:
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Canonical JSONL: sort_keys, (",", ":"), "\\n" per row, utf-8."""
    return b"".join(_row_line(row) for row in rows)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write canonical JSONL exclusively (never overwrites); returns the sha256 of the bytes."""
    path = guards.refuse_sealed_path(path)
    data = jsonl_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
