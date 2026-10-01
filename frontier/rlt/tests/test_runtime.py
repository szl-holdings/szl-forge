"""Negative paths and real tensor/byte roundtrips, with no provider connection."""
from dataclasses import replace
import json
import struct

import pytest
from safetensors.torch import load, save
import torch

from frontier.rlt.model import TinyRLT
from frontier.rlt.runtime import ContinuityError, Scope, Session, canonical, digest


def fixtures(window=4):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    scope = Scope(**{k: digest(k.encode()) for k in Scope.__dataclass_fields__})
    return TinyRLT(window=window), scope


@pytest.mark.parametrize("window", [1, 2, 4])
def test_checkpoint_complete_state_roundtrip(window):
    model, scope = fixtures(window)
    session = Session(model, scope)
    session.append([1, 2, 3])
    artifact = session.checkpoint()
    restored = Session.restore(model, scope, artifact, expected_sha256=digest(artifact), consumed_tokens=[1, 2, 3])
    torch.testing.assert_close(session.append([4, 5]), restored.append([4, 5]), rtol=0, atol=0)
    assert session.observation() == restored.observation()


@pytest.mark.parametrize("field", list(Scope.__dataclass_fields__))
def test_each_scope_axis_requires_replay(field):
    model, scope = fixtures()
    session = Session(model, scope)
    session.append([1, 2])
    artifact = session.checkpoint()
    changed = replace(scope, **{field: digest(b"changed")})
    with pytest.raises(ContinuityError):
        Session.restore(model, changed, artifact, expected_sha256=digest(artifact), consumed_tokens=[1, 2])
    assert session.replay(changed, [1, 2]).observation()["identity_sha256"] != session.observation()["identity_sha256"]


def test_pending_is_consumed_once_across_checkpoint():
    model, scope = fixtures()
    session = Session(model, scope)
    session.append([1, 2])
    pending = session.emit_greedy()
    artifact = session.checkpoint()
    with pytest.raises(ContinuityError):
        session.emit_greedy()
    with pytest.raises(ContinuityError):
        session.append([(pending + 1) % model.vocab])
    restored = Session.restore(model, scope, artifact, expected_sha256=digest(artifact),
                               consumed_tokens=[1, 2], pending_token=pending)
    torch.testing.assert_close(session.append([pending]), restored.append([pending]), rtol=0, atol=0)
    assert restored.consumed == 3


def test_corruption_wrong_digest_and_changed_history_rejected():
    model, scope = fixtures()
    session = Session(model, scope)
    session.append([1, 2, 3])
    artifact = session.checkpoint()
    for history in ([1, 2], [1, 9, 3], [1, 2, 3, 4]):
        with pytest.raises(ContinuityError):
            Session.restore(model, scope, artifact, expected_sha256=digest(artifact), consumed_tokens=history)
    with pytest.raises(ContinuityError):
        Session.restore(model, scope, artifact[:-1] + bytes([artifact[-1] ^ 1]),
                        expected_sha256=digest(artifact), consumed_tokens=[1, 2, 3])


def test_loaded_weight_mutation_invalidates_cached_state():
    model, scope = fixtures()
    session = Session(model, scope)
    session.append([1])
    with torch.no_grad():
        session._model.embedding.weight[1].add_(1)
    with pytest.raises(ContinuityError):
        session.append([2])


def test_external_model_is_not_the_owned_session_model():
    model, scope = fixtures()
    session = Session(model, scope)
    identity = session.observation()["identity_sha256"]
    with torch.no_grad():
        model.embedding.weight[1].add_(1)
    assert session.observation()["identity_sha256"] == identity


def test_mutated_tensor_or_execution_mode_invalidates_state():
    model, scope = fixtures()
    session = Session(model, scope)
    session.append([1])
    session._state.output.add_(1)
    with pytest.raises(ContinuityError):
        session.checkpoint()
    with torch.autocast("cpu"):
        with pytest.raises(ContinuityError):
            Session(model, scope)


@pytest.mark.parametrize("values", [[], [True], [-1], [32], [1.0], [1, 2, -1], [1] * 129])
def test_invalid_append_does_not_partially_commit(values):
    model, scope = fixtures()
    session = Session(model, scope)
    before = session.observation()
    with pytest.raises(ContinuityError):
        session.append(values)
    assert session.observation() == before


def test_checkpoint_without_any_context_and_single_flight():
    model, scope = fixtures()
    session = Session(model, scope)
    artifact = session.checkpoint()
    restored = Session.restore(model, scope, artifact, expected_sha256=digest(artifact), consumed_tokens=[])
    torch.testing.assert_close(session.append([1]), restored.append([1]), rtol=0, atol=0)
    assert session._lock.acquire(blocking=False)
    try:
        with pytest.raises(ContinuityError):
            session.append([2])
    finally:
        session._lock.release()


@pytest.mark.parametrize("kind", ["missing", "extra", "shape", "nonfinite", "purpose", "bool-count"])
def test_invalid_structures_fail_even_with_matching_external_digest(kind):
    model, scope = fixtures()
    session = Session(model, scope)
    session.append([1, 2])
    artifact = session.checkpoint()
    size = struct.unpack("<Q", artifact[:8])[0]
    metadata = json.loads(artifact[8:8 + size])["__metadata__"]
    tensors = load(artifact)
    if kind == "missing":
        del tensors["encoder_k"]
    elif kind == "extra":
        tensors["unwanted"] = torch.ones(1)
    elif kind == "shape":
        tensors["decoder_v"] = torch.zeros(1)
    elif kind == "nonfinite":
        tensors["output"].fill_(float("nan"))
    else:
        passport = json.loads(metadata["passport"])
        passport["purpose" if kind == "purpose" else "consumed"] = "TRAINING" if kind == "purpose" else True
        metadata["passport"] = canonical(passport).decode()
    bad = save(tensors, metadata=metadata)
    with pytest.raises(ContinuityError):
        Session.restore(model, scope, bad, expected_sha256=digest(bad), consumed_tokens=[1, 2])


def test_projection_contains_only_sanitized_observation():
    model, scope = fixtures()
    session = Session(model, scope)
    session.append([1, 2, 3])
    view = session.observation()
    assert view["task_verified"] is False and view["execution_authority"] == "NONE"
    assert not {"tokens", "prompt", "tensor", "hidden_state", "chain_of_thought"} & view.keys()
    json.dumps(view, allow_nan=False)
