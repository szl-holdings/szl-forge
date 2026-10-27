"""Pure proposal state for development-loss stopping; no training/authority API.

Callers supply decimal strings and checkpoint identities, persist returned bytes
with the completed optimizer checkpoint, and pin the latest state hash externally.
This module performs no I/O and does not attest checkpoint bytes or metric truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, localcontext
import hashlib
import json
import re

SCHEMA = "szl.development-early-stop-proposal/v1"
MAX_STEP = (1 << 63) - 1
MAX_STATE_BYTES = 16384
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")


class StopStateError(ValueError):
    pass


def _require(ok, code):
    if not ok:
        raise StopStateError(code)


def _integer(value, minimum, maximum, code):
    _require(type(value) is int and minimum <= value <= maximum, code)


def _identity(value, code):
    _require(type(value) is str and _IDENTITY.fullmatch(value) is not None, code)


def _hash(value, code):
    _require(type(value) is str and _HASH.fullmatch(value) is not None, code)


def decimal_text(value):
    """Accept bounded plain decimal strings; canonicalize zeros/trailing zeros.

    Floats, Decimal objects, exponent notation, bools, and nonfinite values are
    deliberately rejected at this boundary. At most 64 digits are permitted.
    """
    _require(type(value) is str and len(value) <= 66, "DECIMAL_STRING_REQUIRED")
    _require(_DECIMAL.fullmatch(value) is not None, "INVALID_DECIMAL")
    _require(sum(c.isdigit() for c in value) <= 64, "DECIMAL_DIGIT_LIMIT")
    parsed = Decimal(value)
    _require(parsed.is_finite(), "NONFINITE_METRIC")
    if parsed == 0:
        return "0"
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


@dataclass(frozen=True)
class Config:
    candidate_id: str
    run_id: str
    development_sha256: str
    evaluation_interval: int
    direction: str
    min_delta: str
    patience: int

    def __post_init__(self):
        _identity(self.candidate_id, "INVALID_CANDIDATE_ID")
        _identity(self.run_id, "INVALID_RUN_ID")
        _hash(self.development_sha256, "INVALID_DEVELOPMENT_HASH")
        _integer(self.evaluation_interval, 1, 1_000_000_000, "INVALID_INTERVAL")
        _require(type(self.direction) is str and self.direction == "min", "DIRECTION_MIN_REQUIRED")
        normalized = decimal_text(self.min_delta)
        _require(Decimal(normalized) >= 0, "NEGATIVE_MIN_DELTA")
        object.__setattr__(self, "min_delta", normalized)
        _integer(self.patience, 1, 1_000_000, "INVALID_PATIENCE")


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    sha256: str

    def __post_init__(self):
        _identity(self.checkpoint_id, "INVALID_CHECKPOINT_ID")
        _hash(self.sha256, "INVALID_CHECKPOINT_HASH")


@dataclass(frozen=True)
class State:
    config: Config
    evaluations: int = 0
    last_step: int = 0
    last_metric: str | None = None
    last_checkpoint: Checkpoint | None = None
    best_step: int | None = None
    best_metric: str | None = None
    best_checkpoint: Checkpoint | None = None
    bad_evaluations: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    def __post_init__(self):
        _validate_state(self)


def _improved(metric, best, delta):
    # Bounded input has <=64 digits; 256 accommodates subtraction across the
    # full permitted integer/fraction range independently of ambient context.
    with localcontext(Context(prec=256, Emax=999999, Emin=-999999)):
        return Decimal(metric) < Decimal(best) - Decimal(delta)


def _validate_state(state):
    _require(type(state) is State and type(state.config) is Config, "STATE_CONFIG_REQUIRED")
    config = state.config
    # Reconstruct to reject any object altered outside frozen dataclass APIs.
    _require(Config(**_config_dict(config)) == config, "INVALID_CONFIG")
    _integer(state.evaluations, 0, MAX_STEP, "INVALID_EVALUATIONS")
    _integer(state.last_step, 0, MAX_STEP, "INVALID_LAST_STEP")
    _integer(state.bad_evaluations, 0, config.patience, "INVALID_BAD_EVALUATIONS")
    _require(type(state.stopped) is bool, "STOPPED_BOOL_REQUIRED")
    _require(state.last_step == state.evaluations * config.evaluation_interval, "INVALID_CADENCE")
    _require(state.stopped == (state.bad_evaluations == config.patience), "INVALID_STOP_LATCH")
    _require(state.stop_reason == ("PATIENCE_EXHAUSTED" if state.stopped else None), "INVALID_STOP_REASON")
    if state.evaluations == 0:
        _require(state.bad_evaluations == 0 and all(x is None for x in (
            state.last_metric, state.last_checkpoint, state.best_step,
            state.best_metric, state.best_checkpoint)), "INVALID_EMPTY_STATE")
        return
    _integer(state.best_step, config.evaluation_interval, state.last_step, "INVALID_BEST_STEP")
    _require(state.best_step % config.evaluation_interval == 0, "INVALID_BEST_CADENCE")
    _require(state.bad_evaluations == state.evaluations - state.best_step // config.evaluation_interval,
             "PATIENCE_HISTORY_MISMATCH")
    for metric in (state.last_metric, state.best_metric):
        _require(decimal_text(metric) == metric, "NONCANONICAL_METRIC")
    for checkpoint in (state.last_checkpoint, state.best_checkpoint):
        _require(type(checkpoint) is Checkpoint, "CHECKPOINT_REQUIRED")
        Checkpoint(checkpoint.checkpoint_id, checkpoint.sha256)
    if state.bad_evaluations == 0:
        _require(state.last_metric == state.best_metric and state.last_checkpoint == state.best_checkpoint,
                 "BEST_IDENTITY_MISMATCH")
    else:
        _require(not _improved(state.last_metric, state.best_metric, config.min_delta),
                 "UNRECORDED_IMPROVEMENT")


def new_state(config):
    """Create candidate state only; no run, ledger, or training is created."""
    return State(config=config)


def observe(state, *, config, step, metric, checkpoint):
    """Record the next scheduled development evaluation, or fail closed.

    The first evaluation must be at evaluation_interval; each later evaluation
    advances exactly one interval. Improvement is STRICTLY greater than delta.
    A stopped state cannot consume another evaluation or clear its stop latch.
    """
    _validate_state(state)
    _require(type(config) is Config and config == state.config, "CONFIG_BINDING_MISMATCH")
    _require(not state.stopped, "EVALUATION_AFTER_STOP")
    _integer(step, 1, MAX_STEP, "INVALID_STEP")
    _require(step == state.last_step + config.evaluation_interval, "EVALUATION_STEP_MISMATCH")
    normalized = decimal_text(metric)
    _require(type(checkpoint) is Checkpoint, "CHECKPOINT_REQUIRED")
    improved = state.best_metric is None or _improved(normalized, state.best_metric, config.min_delta)
    bad = 0 if improved else state.bad_evaluations + 1
    stopped = bad == config.patience
    return State(config=config, evaluations=state.evaluations + 1, last_step=step,
                 last_metric=normalized, last_checkpoint=checkpoint,
                 best_step=step if improved else state.best_step,
                 best_metric=normalized if improved else state.best_metric,
                 best_checkpoint=checkpoint if improved else state.best_checkpoint,
                 bad_evaluations=bad, stopped=stopped,
                 stop_reason="PATIENCE_EXHAUSTED" if stopped else None)


def _config_dict(config):
    return {name: getattr(config, name) for name in (
        "candidate_id", "run_id", "development_sha256", "evaluation_interval",
        "direction", "min_delta", "patience")}


def _checkpoint_dict(checkpoint):
    if checkpoint is None:
        return None
    return {"checkpoint_id": checkpoint.checkpoint_id, "sha256": checkpoint.sha256}


def canonical_bytes(state):
    """Encode exact state bytes; caller owns atomic persistence and fresh pins."""
    _validate_state(state)
    document = {
        "schema": SCHEMA, "config": _config_dict(state.config),
        "evaluations": state.evaluations, "last_step": state.last_step,
        "last_metric": state.last_metric, "last_checkpoint": _checkpoint_dict(state.last_checkpoint),
        "best_step": state.best_step, "best_metric": state.best_metric,
        "best_checkpoint": _checkpoint_dict(state.best_checkpoint),
        "bad_evaluations": state.bad_evaluations, "stopped": state.stopped,
        "stop_reason": state.stop_reason,
    }
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def state_sha256(state):
    return hashlib.sha256(canonical_bytes(state)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _reject_number(value):
    raise StopStateError("JSON_DECIMAL_STRING_REQUIRED")


def _parse_int(value):
    _require(len(value) <= 20, "JSON_INTEGER_LIMIT")
    return int(value)


def _fields(value, names, code):
    _require(type(value) is dict and set(value) == set(names), code)


def _parse_checkpoint(value):
    if value is None:
        return None
    _fields(value, ("checkpoint_id", "sha256"), "CHECKPOINT_FIELDS")
    return Checkpoint(**value)


def restore(raw, *, config, expected_state_sha256):
    """Restore against an independently retained latest hash and frozen config.

    Caller must select a fresh trusted hash, never derive that expected pin from
    untrusted saved bytes. A hash does not prove freshness, metric truth, or authority.
    """
    _require(type(raw) is bytes and 2 <= len(raw) <= MAX_STATE_BYTES, "STATE_BYTES_LIMIT")
    _require(type(config) is Config, "CONFIG_REQUIRED")
    _hash(expected_state_sha256, "EXPECTED_STATE_HASH_REQUIRED")
    _require(hashlib.sha256(raw).hexdigest() == expected_state_sha256, "STATE_HASH_MISMATCH")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                              parse_float=_reject_number, parse_constant=_reject_number,
                              parse_int=_parse_int)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise StopStateError("INVALID_STATE_JSON") from error
    _fields(document, ("schema", "config", "evaluations", "last_step", "last_metric",
                      "last_checkpoint", "best_step", "best_metric", "best_checkpoint",
                      "bad_evaluations", "stopped", "stop_reason"), "STATE_FIELDS")
    _require(document["schema"] == SCHEMA, "STATE_SCHEMA_MISMATCH")
    _fields(document["config"], _config_dict(config), "CONFIG_FIELDS")
    restored_config = Config(**document["config"])
    _require(restored_config == config, "CONFIG_BINDING_MISMATCH")
    values = {name: document[name] for name in (
        "evaluations", "last_step", "last_metric", "best_step", "best_metric",
        "bad_evaluations", "stopped", "stop_reason")}
    state = State(config=config, last_checkpoint=_parse_checkpoint(document["last_checkpoint"]),
                  best_checkpoint=_parse_checkpoint(document["best_checkpoint"]), **values)
    _require(canonical_bytes(state) == raw, "NONCANONICAL_STATE_JSON")
    return state


if __name__ == "__main__":
    raise SystemExit("BLOCKED: source proposal only; no trainer or execution authority")

