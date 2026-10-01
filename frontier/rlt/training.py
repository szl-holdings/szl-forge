"""Bounded register experiment: real synthetic training, not an LLM release.

SET0/SET1/FLIP/KEEP/REVOKE update a register or UNKNOWN. A separate deterministic
oracle creates labels. Full gradients cross context positions. Held-out programs
never choose checkpoints or hyperparameters. No external data/network/publication.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import statistics
import time

from safetensors.torch import load, save
import torch
from torch import nn

from .model import TinyRLT
from .runtime import ContinuityError, Scope, Session, canonical, digest

TASK = "szl.synthetic-revision-register.v1"
OPERATIONS = ("SET0", "SET1", "FLIP", "KEEP", "REVOKE")
VARIANTS = ("rlt", "no_output_feedback", "causal_transformer")
LABEL_OFFSET = 8
CONFIG = {"vocab": 32, "dim": 16, "heads": 2, "window": 4, "max_tokens": 128}


def oracle(program: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    state = 2
    result = []
    for op in program:
        if type(op) is not int or not 0 <= op < len(OPERATIONS):
            raise ValueError("invalid synthetic operation")
        if op < 2:
            state = op
        elif op == 2 and state < 2:
            state = 1 - state
        elif op == 4:
            state = 2
        result.append(state)
    return tuple(result)


def split_of(program: tuple[int, ...] | list[int]) -> str:
    """Programs sharing their first six operations always share one split."""
    if len(program) < 6:
        raise ValueError("split assignment needs six operations")
    bucket = int(sha256(bytes(program[:6])).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def programs(split: str, length: int, count: int, seed: int) -> tuple[tuple[int, ...], ...]:
    if split not in {"train", "validation", "test"}:
        raise ValueError("unknown split")
    if type(length) is not int or not 8 <= length <= 64:
        raise ValueError("length must be in [8,64]")
    if type(count) is not int or not 1 <= count <= 2048 or type(seed) is not int:
        raise ValueError("invalid count or seed")
    rng = random.Random(int(sha256(f"{TASK}:{split}:{length}:{seed}".encode()).hexdigest(), 16))
    rows, seen = [], set()
    for _ in range(count * 1000):
        row = tuple(rng.randrange(len(OPERATIONS)) for _ in range(length))
        if split_of(row) == split and row not in seen:
            rows.append(row)
            seen.add(row)
            if len(rows) == count:
                return tuple(rows)
    raise RuntimeError("bounded corpus generation exhausted")


def dataset_bytes(rows: tuple[tuple[int, ...], ...]) -> bytes:
    return b"".join(canonical({"operations": p, "states": oracle(p)}) + b"\n" for p in rows)


class CausalBaseline(nn.Module):
    """Two untied encoder layers, same width/data; NOT parameter/FLOP matched."""
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.position = nn.Embedding(128, 16)
        layer = nn.TransformerEncoderLayer(16, 2, 32, dropout=0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(16, 32)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, None]:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device).triu(1)
        return self.head(self.encoder(hidden, mask=mask)), None


def make_model(variant: str, seed: int) -> nn.Module:
    if variant not in VARIANTS or type(seed) is not int:
        raise ValueError("invalid variant or seed")
    torch.manual_seed(seed)
    model = CausalBaseline() if variant == "causal_transformer" else TinyRLT(**CONFIG)
    if variant == "no_output_feedback":
        with torch.no_grad():
            model.feedback_scale.zero_()
        model.feedback_scale.requires_grad_(False)
    return model


@dataclass(frozen=True)
class Plan:
    steps: int = 120
    train_length: int = 16
    train_count: int = 256
    eval_count: int = 128
    batch_size: int = 16
    learning_rate: float = 0.003
    seeds: tuple[int, ...] = (17, 29, 43)

    def __post_init__(self) -> None:
        for name, low, high in (("steps", 1, 2000), ("train_length", 8, 32),
                               ("train_count", 32, 2048), ("eval_count", 16, 2048),
                               ("batch_size", 1, 64)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if type(self.learning_rate) not in (int, float) or not 0 < self.learning_rate <= 0.02:
            raise ValueError("invalid learning rate")
        if type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= 5 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("require 1..5 distinct seeds")
        if any(type(s) is not int or not 0 <= s < 2**31 for s in self.seeds):
            raise ValueError("invalid seed")


def evaluate(model: nn.Module, rows: tuple[tuple[int, ...], ...]) -> dict[str, object]:
    correct = final_correct = unknown_correct = unknown_count = total = 0
    predictions = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), 32):
            chunk = rows[offset:offset+32]
            x = torch.tensor(chunk, dtype=torch.long)
            y = torch.tensor([oracle(p) for p in chunk], dtype=torch.long)
            logits, _ = model(x)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite evaluation logits")
            pred = logits[..., LABEL_OFFSET:LABEL_OFFSET+3].argmax(-1)
            correct += int((pred == y).sum())
            final_correct += int((pred[:, -1] == y[:, -1]).sum())
            unknown = y == 2
            unknown_count += int(unknown.sum())
            unknown_correct += int(((pred == 2) & unknown).sum())
            total += y.numel()
            predictions.extend(pred[:, -1].tolist())
    return {"programs": len(rows), "positions": total, "correct_positions": correct,
            "final_correct": final_correct, "position_accuracy": correct / total,
            "final_accuracy": final_correct / len(rows), "unknown_positions": unknown_count,
            "unknown_correct": unknown_correct,
            "unknown_recall": unknown_correct / unknown_count if unknown_count else None,
            "prediction_sha256": digest(canonical(predictions)), "wall_seconds": time.perf_counter()-start}


def weight_bytes(model: nn.Module) -> bytes:
    return save({name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()})


def load_candidate(blob: bytes, *, expected_sha256: str) -> TinyRLT:
    """Fixed architecture load; expected digest must be independently trusted."""
    if type(blob) is not bytes or len(blob) > 1024*1024 or digest(blob) != expected_sha256:
        raise ContinuityError("candidate bytes mismatch or resource bound exceeded")
    model = TinyRLT(**CONFIG)
    supplied = load(blob)
    expected = model.state_dict()
    if set(supplied) != set(expected):
        raise ContinuityError("candidate tensor set differs")
    for key, value in supplied.items():
        if value.shape != expected[key].shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ContinuityError("candidate tensor shape/dtype/finiteness differs")
    model.load_state_dict(supplied, strict=True)
    return model.eval()


def one_run(variant: str, seed: int, plan: Plan, train_rows: tuple[tuple[int, ...], ...],
            validation_rows: tuple[tuple[int, ...], ...], tests: dict[int, tuple[tuple[int, ...], ...]]) -> tuple[dict, bytes]:
    model = make_model(variant, seed)
    initial = digest(weight_bytes(model))
    initial_score = evaluate(model, validation_rows)
    generator = torch.Generator().manual_seed(seed+1000)
    x = torch.tensor(train_rows, dtype=torch.long)
    y = torch.tensor([oracle(p) for p in train_rows], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.learning_rate, weight_decay=0.01)
    started = time.perf_counter()
    model.train()
    losses = []
    for _ in range(plan.steps):
        indices = torch.randint(len(train_rows), (plan.batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x[indices])
        loss = nn.functional.cross_entropy(logits[..., LABEL_OFFSET:LABEL_OFFSET+3].reshape(-1, 3), y[indices].reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    train_seconds = time.perf_counter()-started
    model.eval()
    blob = weight_bytes(model)
    validation = evaluate(model, validation_rows)
    test_results = {str(length): evaluate(model, rows) for length, rows in tests.items()}
    report = {"variant": variant, "seed": seed, "steps_completed": len(losses),
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "initial_weight_sha256": initial, "weight_sha256": digest(blob),
              "weight_bytes": len(blob), "loss_first": losses[0], "loss_last": losses[-1],
              "loss_curve_sha256": digest(canonical(losses)), "training_seconds": train_seconds,
              "validation_before": initial_score, "validation_after": validation, "test": test_results}
    if variant == "rlt":
        loaded = load_candidate(blob, expected_sha256=digest(blob))
        with torch.no_grad():
            a, _ = model(x[:2]); b, _ = loaded(x[:2])
        report["weight_roundtrip_max_abs_error"] = float((a-b).abs().max())
        scope = Scope(**{k: digest(k.encode()) for k in Scope.__dataclass_fields__})
        session = Session(loaded, scope)
        history = list(train_rows[0])
        session.append(history[:8])
        snapshot = session.checkpoint()
        resumed = Session.restore(loaded, scope, snapshot, expected_sha256=digest(snapshot), consumed_tokens=history[:8])
        a, b = session.append(history[8:] or [3]), resumed.append(history[8:] or [3])
        report["state_roundtrip_max_abs_error"] = float((a-b).abs().max())
    return report, blob


def experiment(plan: Plan, output: Path) -> dict[str, object]:
    """Write real trained artifacts to a new private local directory only."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_hashes = {p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    datasets = {"train": programs("train", plan.train_length, plan.train_count, 211),
                "validation": programs("validation", plan.train_length, plan.eval_count, 211)}
    tests = {n: programs("test", n, plan.eval_count, 211) for n in sorted({plan.train_length, 32, 64})}
    datasets.update({f"test_{n}": rows for n, rows in tests.items()})
    dataset_manifest = {}
    for name, rows in datasets.items():
        blob = dataset_bytes(rows)
        (output/f"{name}.jsonl").write_bytes(blob)
        dataset_manifest[name] = {"rows": len(rows), "sha256": digest(blob), "bytes": len(blob)}
    results = []
    for seed in plan.seeds:
        for variant in VARIANTS:
            report, blob = one_run(variant, seed, plan, datasets["train"], datasets["validation"], tests)
            filename = f"{variant}-{seed}.safetensors"
            (output/filename).write_bytes(blob)
            report["artifact"] = filename
            results.append(report)
            print(json.dumps({"variant": variant, "seed": seed, "trained": True,
                              "test": {n: m["final_accuracy"] for n,m in report["test"].items()}}), flush=True)
    summary = {variant: {str(n): {"mean": statistics.mean(r["test"][str(n)]["final_accuracy"] for r in results if r["variant"]==variant),
                                 "minimum": min(r["test"][str(n)]["final_accuracy"] for r in results if r["variant"]==variant),
                                 "maximum": max(r["test"][str(n)]["final_accuracy"] for r in results if r["variant"]==variant)}
                          for n in tests} for variant in VARIANTS}
    if source_hashes != {p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}:
        raise RuntimeError("source moved during experiment; no terminal evaluation receipt")
    report = {"schema": "szl.rlt.synthetic-training-evaluation.v1", "task": TASK, "plan": asdict(plan),
              "status": "MEASURED_SYNTHETIC_ONLY", "trained": True, "production_qualified": False,
              "publication_eligible": False, "execution_authority": "NONE", "receipt_status": "UNSIGNED",
              "dataset": dataset_manifest, "runs": results, "summary": summary,
              "environment": {"torch": torch.__version__, "python": platform.python_version(), "device": "cpu", "threads": 1},
              "source_sha256": source_hashes,
              "selection": "All seeds/variants reported; test data do not choose a model or hyperparameters",
              "limits": ["Not natural-language or production training", "Supervised state labels, not language-model next-token training", "Same examples/steps, not parameter/FLOP/time matched",
                         "No GPU/energy/cost-saving claim", "UNKNOWN register state is a synthetic label, not general abstention",
                         "Dataset partitions share a grammar; program-family separation is not semantic independence",
                         "Weights are inference artifacts; optimizer-resume training checkpoint not implemented"]}
    report["receipt_sha256"] = digest(canonical(report))
    (output/"evaluation.json").write_bytes(canonical(report)+b"\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=120, type=int)
    parser.add_argument("--single-seed", type=int)
    args = parser.parse_args()
    plan = Plan(steps=args.steps, seeds=(args.single_seed,) if args.single_seed is not None else (17,29,43))
    report = experiment(plan, args.output)
    print(json.dumps({"status": report["status"], "receipt_sha256": report["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
