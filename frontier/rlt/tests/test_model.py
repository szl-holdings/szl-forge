import pytest
import torch
from frontier.rlt.model import TinyRLT, detached_state


def model(window=4):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    return TinyRLT(window=window).double()


def tokens():
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)


@pytest.mark.parametrize("window", [1, 2, 4])
def test_parallel_encoder_equals_sequential_reference(window):
    m = model(window)
    sequential, _ = m(tokens())
    prefilled, _ = m.parallel_encoder_prefill(tokens())
    torch.testing.assert_close(sequential, prefilled, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("split", [1, 3, 7])
def test_prompt_tool_or_response_split_is_not_a_reset(split):
    m = model()
    expected, _ = m(tokens())
    first, state = m.parallel_encoder_prefill(tokens()[:, :split])
    parts = [first]
    for t in range(split, tokens().shape[1]):
        logits, state = m.step(tokens()[:, t:t+1], state)
        parts.append(logits)
    torch.testing.assert_close(torch.cat(parts, dim=1), expected, atol=1e-10, rtol=1e-10)


def test_future_encoder_tokens_do_not_leak():
    m = model()
    changed = tokens().clone()
    changed[:, 4:] = 21
    left, _ = m.parallel_encoder_prefill(tokens())
    right, _ = m.parallel_encoder_prefill(changed)
    torch.testing.assert_close(left[:, :4], right[:, :4], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("window", [1, 2, 4])
def test_complete_state_cache_lengths(window):
    m = model(window)
    state = m.empty_state(1)
    for t in range(tokens().shape[1]):
        _, state = m.step(tokens()[:, t:t+1], state)
        assert state.consumed == t + 1
        assert state.decoder_kv.k.shape[2] == min(t + 1, window - 1)
        assert state.encoder_kv.k.shape[2] == t + 1
        assert state.encoder_memory.k.shape[2] == t + 1


def test_full_bptt_reaches_initial_state_and_early_context():
    m = model()
    logits, _ = m(tokens())
    # Supervise only the last target; earlier context still participates in gradients.
    torch.nn.functional.cross_entropy(logits[:, -1], torch.tensor([9])).backward()
    assert m.initial.grad is not None and m.initial.grad.abs().sum() > 0
    assert m.embedding.weight.grad[1].abs().sum() > 0
    assert m.shared_attention.k.weight.grad.abs().sum() > 0
    assert m.cross_attention.k.weight.grad.abs().sum() > 0


def test_detaching_prefix_removes_early_context_gradient():
    m = model()
    _, state = m(tokens()[:, :4])
    state = detached_state(state)
    for t in range(4, 8):
        logits, state = m.step(tokens()[:, t:t+1], state)
    logits.square().sum().backward()
    assert torch.count_nonzero(m.embedding.weight.grad[1]) == 0


def test_changed_weights_make_old_snapshot_numerically_stale():
    m = model()
    _, old = m(tokens()[:, :4])
    old = detached_state(old)
    with torch.no_grad():
        m.embedding.weight[1].add_(2.0)
    _, fresh = m(tokens()[:, :4])
    stale_logits, _ = m.step(tokens()[:, 4:5], old)
    fresh_logits, _ = m.step(tokens()[:, 4:5], fresh)
    assert not torch.allclose(stale_logits, fresh_logits, atol=1e-8, rtol=1e-8)


def test_independent_batch_rows_do_not_share_state():
    m = model()
    a, b = tokens(), tokens().flip(dims=[1])
    batch, _ = m(torch.cat((a, b), dim=0))
    left, _ = m(a)
    right, _ = m(b)
    torch.testing.assert_close(batch[:1], left, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(batch[1:], right, atol=1e-10, rtol=1e-10)


def test_context_overflow_is_error():
    m = model()
    with pytest.raises(ValueError):
        m(torch.ones(1, m.max_tokens + 1, dtype=torch.long))
