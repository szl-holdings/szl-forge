"""Small trainable baselines, intentionally not foundation models or hard gates."""
from __future__ import annotations
import math
from collections.abc import Mapping
import torch
from torch import nn
from .catalog import track_for

class AdvisoryMLP(nn.Module):
    """An 8→16→1 MLP returning a logit; sigmoid output is NOT calibrated trust."""
    def __init__(self, track: str):
        super().__init__()
        self.track = track_for(track)
        self.network = nn.Sequential(nn.Linear(len(self.track.features), 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != len(self.track.features):
            raise ValueError("invalid_feature_tensor_shape")
        if x.device.type != "cpu":
            raise ValueError("baseline_cpu_only")
        if not torch.isfinite(x).all() or torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("invalid_feature_tensor_values")
        return self.network(x).squeeze(-1)

    def vector(self, features: Mapping[str, float]) -> torch.Tensor:
        if set(features) != set(self.track.features):
            raise ValueError("feature_schema_mismatch")
        values = [features[n] for n in self.track.features]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("invalid_feature_values")
        return torch.tensor([values], dtype=torch.float32)

    def score(self, features: Mapping[str, float]) -> float:
        self.eval()
        with torch.inference_mode():
            result = float(torch.sigmoid(self(self.vector(features)))[0])
        if not math.isfinite(result):
            raise ValueError("nonfinite_model_output")
        return result

def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Count-preserving binary metrics; threshold fixed in advance at 0.5."""
    if logits.ndim != 1 or labels.shape != logits.shape or len(labels) == 0:
        raise ValueError("invalid_metric_shape")
    if not torch.isfinite(logits).all() or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("invalid_metric_values")
    p = torch.sigmoid(logits)
    pred = p >= 0.5
    positive = labels == 1
    return {"n": len(labels), "positives": int(positive.sum()),
            "true_positives": int((pred & positive).sum()),
            "false_positives": int((pred & ~positive).sum()),
            "true_negatives": int((~pred & ~positive).sum()),
            "false_negatives": int((~pred & positive).sum()),
            "brier_score": float(((p - labels) ** 2).mean()),
            "log_loss": float(nn.functional.binary_cross_entropy_with_logits(logits, labels)),
            "threshold": 0.5, "calibration_validated": False,
            "generalization_claim": "NONE"}
