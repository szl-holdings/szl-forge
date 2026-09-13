"""Small trainable counterparts, separate from deterministic SZL kernels.

All factories initialize NEW RANDOM parameters. This module never loads remote
code, downloads a base, starts a job, routes requests, or publishes artifacts.
Loss functions are minibatch interfaces for a future reviewed supervisor adapter,
not a second trainer. Output distributions are not calibrated confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .specs import FEATURES, RISK_LABELS, candidate_spec


def _features(value: Tensor, ndim: int) -> None:
    if value.ndim != ndim or value.shape[-1] != len(FEATURES):
        raise ValueError("feature shape or ordered schema mismatch")
    if any(size < 1 for size in value.shape) or value.shape[0] > 32:
        raise ValueError("empty or oversized batch")
    if ndim == 3 and value.shape[1] > 64:
        raise ValueError("at most 64 candidate routes")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError("features must be finite floats")
    if not ((value >= 0) & (value <= 1)).all():
        raise ValueError("features must be explicitly normalized into [0,1]")


def _mlp(key: str, outputs: int) -> nn.Sequential:
    config = candidate_spec(key)["config"]
    width = config["hidden_dim"]
    return nn.Sequential(
        nn.Linear(config["input_dim"], width), nn.GELU(),
        nn.Linear(width, width), nn.GELU(), nn.Linear(width, outputs),
    )


@dataclass(frozen=True)
class RouteProposal:
    logits: Tensor
    distribution: Tensor
    eligible: Tensor
    selected_index: Tensor  # -1 means ABSTAIN; never a dispatch instruction.


class RouteUtilityModel(nn.Module):
    """Shared scorer across a variable set of externally eligible routes."""

    def __init__(self) -> None:
        super().__init__()
        self.scorer = _mlp("router", 1)

    def forward(self, features: Tensor, eligible: Tensor) -> RouteProposal:
        _features(features, 3)
        if eligible.dtype != torch.bool or eligible.shape != features.shape[:2]:
            raise ValueError("eligibility must be an exact boolean B x R mask")
        if eligible.device != features.device:
            raise ValueError("eligibility and features must share a device")
        logits = self.scorer(features).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite learned route scores")
        available = eligible.any(dim=-1)
        masked = logits.masked_fill(~eligible, float("-inf"))
        # Avoid all-minus-infinity softmax: unavailable rows return zero mass.
        safe = torch.where(available[:, None], masked, torch.zeros_like(masked))
        distribution = safe.softmax(dim=-1) * eligible.to(logits.dtype)
        selected = distribution.argmax(dim=-1)
        selected = torch.where(available, selected, torch.full_like(selected, -1))
        return RouteProposal(logits, distribution, eligible.clone(), selected)


def route_loss(proposal: RouteProposal, winners: Tensor) -> Tensor:
    """Supervised ranking loss; labels MUST be observed eligible winners.

The data admission lane must establish label provenance and counterfactual
coverage. This function validates tensors, not the truth of human labels.
"""
    batch, routes = proposal.logits.shape
    if winners.dtype != torch.long or winners.shape != (batch,):
        raise ValueError("winner labels must be int64 B")
    if winners.device != proposal.logits.device:
        raise ValueError("labels and scores must share a device")
    if not ((winners >= 0) & (winners < routes)).all():
        raise ValueError("winner out of range")
    if not proposal.eligible.gather(1, winners[:, None]).all():
        raise ValueError("training label selects an ineligible route")
    return F.cross_entropy(
        proposal.logits.masked_fill(~proposal.eligible, float("-inf")), winners,
    )


class InvariantRiskModel(nn.Module):
    """Four advisory outcome-risk logits; not a verifier or authorization gate."""

    def __init__(self) -> None:
        super().__init__()
        self.scorer = _mlp("invariant-risk", len(RISK_LABELS))

    def forward(self, features: Tensor) -> Tensor:
        _features(features, 2)
        logits = self.scorer(features)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite learned risk scores")
        return logits


def risk_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] != len(RISK_LABELS) or logits.shape[0] < 1:
        raise ValueError("risk logits must be B x 4")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("risk logits must be finite floats")
    if labels.shape != logits.shape or labels.device != logits.device:
        raise ValueError("risk label shape/device mismatch")
    if not labels.is_floating_point() or not torch.isfinite(labels).all():
        raise ValueError("risk labels must be finite floats")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("risk labels must be observed binary outcomes")
    return F.binary_cross_entropy_with_logits(logits, labels)


def canal_mask(length: int, canal_width: int, device: torch.device) -> Tensor:
    """True means attend. Fixed-width canals preserve prefix consistency.

Partitioning by a fixed NUMBER of canals as sequence length grows can change
past boundaries. This research adaptation instead fixes the width and adds a
causal triangle. It is deliberately NOT the canonical noncausal YARQA operator.
"""
    if type(length) is not int or not 1 <= length <= 128:
        raise ValueError("sequence length must be an integer in [1,128]")
    if type(canal_width) is not int or not 1 <= canal_width <= 128:
        raise ValueError("canal width must be an integer in [1,128]")
    position = torch.arange(length, device=device)
    same = (position[:, None] // canal_width) == (position[None, :] // canal_width)
    causal = position[None, :] <= position[:, None]
    return same & causal


class _CanalBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.width, self.heads = width, heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        batch, length, _ = hidden.shape
        qkv = self.qkv(self.norm_attention(hidden))
        q, k, v = qkv.reshape(batch, length, 3, self.heads, self.width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        hidden = hidden + self.output(attended.transpose(1, 2).reshape(batch, length, self.width))
        return hidden + self.ff(self.norm_ff(hidden))


class CanalLanguageModel(nn.Module):
    """Dense causal reference: byte IDs 0..255 plus BOS=256.

This intentionally small two-layer model has no cross-canal context, KV cache,
GGUF converter, Ollama loader or accelerated-kernel claim. Test CPU correctness
first. GPU behavior is unqualified until measured separately.
"""

    def __init__(self) -> None:
        super().__init__()
        config = candidate_spec("yarqa-causal")["config"]
        self.vocab_size = config["vocab_size"]
        self.max_length = config["max_length"]
        self.canal_width = config["canal_width"]
        width, heads = config["width"], config["heads"]
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.position_embedding = nn.Embedding(self.max_length, width)
        self.blocks = nn.ModuleList([_CanalBlock(width, heads) for _ in range(config["layers"])])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, self.vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype != torch.long or token_ids.ndim != 2:
            raise ValueError("token IDs must be int64 B x T")
        batch, length = token_ids.shape
        if not 1 <= batch <= 32 or not 1 <= length <= self.max_length:
            raise ValueError("token batch/length outside research limits")
        if not ((token_ids >= 0) & (token_ids < self.vocab_size)).all():
            raise ValueError("token ID outside byte/BOS vocabulary")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        mask = canal_mask(length, self.canal_width, token_ids.device)
        for block in self.blocks:
            hidden = block(hidden, mask)
        logits = self.lm_head(self.norm(hidden))
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite language-model logits")
        return logits

    def loss(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] < 2:
            raise ValueError("next-token training needs at least two tokens")
        logits = self(token_ids)
        return F.cross_entropy(logits[:, :-1].reshape(-1, self.vocab_size), token_ids[:, 1:].reshape(-1))


def build_candidate(key: str) -> nn.Module:
    factories = {
        "router": RouteUtilityModel,
        "invariant-risk": InvariantRiskModel,
        "yarqa-causal": CanalLanguageModel,
    }
    if key not in factories:
        raise ValueError("unknown candidate")
    return factories[key]()
