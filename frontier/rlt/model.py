"""Untrained, original PyTorch CPU reference inspired by Zhang's RLT report.

Deliberate scope: one encoder block, one recurrent decoder block, multi-head
attention, shared self-attention/FFN weights, learned absolute positions. This is
NOT the paper's 48+48 configuration, a pretrained model, a serving integration,
or evidence of reasoning/efficiency gains. It is a testable execution reference.

No external code or weights are downloaded. No network or tool execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class KV:
    k: Tensor
    v: Tensor


@dataclass(frozen=True)
class CompleteState:
    encoder_kv: KV
    encoder_memory: KV
    decoder_kv: KV
    output: Tensor
    consumed: int


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * self.weight


class AttentionWeights(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)

    def project_kv(self, x: Tensor) -> KV:
        return KV(self.split(self.k(x)), self.split(self.v(x)))

    def attend(self, x: Tensor, memory: KV, *, causal: bool = False) -> Tensor:
        q = self.split(self.q(x))
        scores = q @ memory.k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if causal:
            if q.shape[-2] != memory.k.shape[-2]:
                raise ValueError("batch causal attention requires equal query/key lengths")
            size = q.shape[-2]
            mask = torch.ones(size, size, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        values = torch.softmax(scores, dim=-1) @ memory.v
        values = values.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], self.dim)
        return self.o(values)


def append_kv(old: KV, new: KV) -> KV:
    return KV(torch.cat((old.k, new.k), dim=2), torch.cat((old.v, new.v), dim=2))


def prefix_kv(kv: KV, count: int) -> KV:
    return KV(kv.k[:, :, :count], kv.v[:, :, :count])


def tail_kv(kv: KV, count: int) -> KV:
    # [:0], not [-0:], is essential to the W=1 boundary convention.
    return KV(kv.k[:, :, -count:] if count else kv.k[:, :, :0],
              kv.v[:, :, -count:] if count else kv.v[:, :, :0])


class TinyRLT(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 16, heads: int = 2,
                 window: int = 4, max_tokens: int = 128) -> None:
        super().__init__()
        for name, value in (("vocab", vocab), ("dim", dim), ("heads", heads),
                            ("window", window), ("max_tokens", max_tokens)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.vocab, self.dim, self.window, self.max_tokens = vocab, dim, window, max_tokens
        self.embedding = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(max_tokens, dim)
        # Exactly one set reused by the encoder and decoder.
        self.shared_attention = AttentionWeights(dim, heads)
        self.shared_ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.cross_attention = AttentionWeights(dim, heads)
        self.enc_attention_norm, self.enc_ffn_norm = RMSNorm(dim), RMSNorm(dim)
        self.dec_attention_norm, self.dec_ffn_norm = RMSNorm(dim), RMSNorm(dim)
        self.cross_norm, self.feedback_norm, self.output_norm = RMSNorm(dim), RMSNorm(dim), RMSNorm(dim)
        self.gate = nn.Linear(dim * 2, dim)
        self.feedback = nn.Linear(dim, dim, bias=False)
        self.feedback_scale = nn.Parameter(torch.tensor(0.1))
        self.initial = nn.Parameter(torch.zeros(dim))
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def empty_state(self, batch: int) -> CompleteState:
        weight = self.embedding.weight
        shape = (batch, self.shared_attention.heads, 0, self.shared_attention.head_dim)
        empty = KV(weight.new_empty(shape), weight.new_empty(shape))
        return CompleteState(empty, empty, empty, self.initial.expand(batch, -1).unsqueeze(1), 0)

    def _check_tokens(self, tokens: Tensor) -> None:
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a nonempty [batch, time] torch.long tensor")
        if bool(torch.any(tokens < 0)) or bool(torch.any(tokens >= self.vocab)):
            raise ValueError("token outside vocabulary")
        if tokens.shape[1] > self.max_tokens:
            raise ValueError("context budget exceeded")

    def _decode(self, encoded: Tensor, state: CompleteState, encoder_kv: KV,
                memory: KV) -> tuple[Tensor, CompleteState]:
        previous = self.feedback_norm(state.output)
        gate = torch.sigmoid(self.gate(torch.cat((encoded, previous), dim=-1)))
        hidden = encoded + self.feedback_scale * gate * self.feedback(previous)
        normalized = self.dec_attention_norm(hidden)
        full_local = append_kv(state.decoder_kv, self.shared_attention.project_kv(normalized))
        hidden = hidden + self.shared_attention.attend(normalized, full_local)
        hidden = hidden + self.cross_attention.attend(self.cross_norm(hidden), memory)
        hidden = hidden + self.shared_ffn(self.dec_ffn_norm(hidden))
        next_state = CompleteState(encoder_kv, memory, tail_kv(full_local, self.window - 1),
                                   hidden, state.consumed + 1)
        logits = self.lm_head(self.output_norm(hidden))
        return logits, next_state

    def step(self, token: Tensor, state: CompleteState) -> tuple[Tensor, CompleteState]:
        """Consume one observed token; returned logits predict the NEXT token.

        This low-level math primitive is not a secure cache API. Callers must
        enforce continuity.py and validate actual tensor shapes/dtypes/devices.
        """
        self._check_tokens(token)
        if token.shape[1] != 1 or token.shape[0] != state.output.shape[0]:
            raise ValueError("step expects one token per state batch row")
        if state.consumed >= self.max_tokens:
            raise ValueError("context budget exceeded")
        position = torch.tensor([state.consumed], device=token.device)
        hidden = self.embedding(token) + self.position(position)
        normalized = self.enc_attention_norm(hidden)
        encoder_kv = append_kv(state.encoder_kv, self.shared_attention.project_kv(normalized))
        hidden = hidden + self.shared_attention.attend(normalized, encoder_kv)
        encoded = hidden + self.shared_ffn(self.enc_ffn_norm(hidden))
        memory = append_kv(state.encoder_memory, self.cross_attention.project_kv(encoded))
        return self._decode(encoded, state, encoder_kv, memory)

    def forward(self, tokens: Tensor) -> tuple[Tensor, CompleteState]:
        """Sequential reference; preserves the full autograd graph across tokens."""
        self._check_tokens(tokens)
        state = self.empty_state(tokens.shape[0])
        outputs = []
        for t in range(tokens.shape[1]):
            logits, state = self.step(tokens[:, t:t+1], state)
            outputs.append(logits)
        return torch.cat(outputs, dim=1), state

    def parallel_encoder_prefill(self, tokens: Tensor) -> tuple[Tensor, CompleteState]:
        """Parallel causal encoder, sequential prefix-restricted recurrent decoder."""
        self._check_tokens(tokens)
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        normalized = self.enc_attention_norm(hidden)
        encoder_kv = self.shared_attention.project_kv(normalized)
        hidden = hidden + self.shared_attention.attend(normalized, encoder_kv, causal=True)
        encoded = hidden + self.shared_ffn(self.enc_ffn_norm(hidden))
        memory = self.cross_attention.project_kv(encoded)
        state = self.empty_state(tokens.shape[0])
        outputs = []
        for t in range(length):
            logits, state = self._decode(encoded[:, t:t+1], state,
                                         prefix_kv(encoder_kv, t + 1), prefix_kv(memory, t + 1))
            outputs.append(logits)
        return torch.cat(outputs, dim=1), state


def detached_state(state: CompleteState) -> CompleteState:
    """Inference snapshot values only: intentionally NOT full-gradient training."""
    def detach(kv: KV) -> KV:
        return KV(kv.k.detach().clone(), kv.v.detach().clone())
    return CompleteState(detach(state.encoder_kv), detach(state.encoder_memory),
                         detach(state.decoder_kv), state.output.detach().clone(), state.consumed)
