"""Model-specific CPU research recipes under Forge; not a new job supervisor.

The web app never imports this module or starts training. CLI training requires
an explicit acknowledgement and a clean, exact GitHub-source checkout. Existing
Owned Agent Control / gpu-bridge remain the eventual execution authorities.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import subprocess
from pathlib import Path
import torch
from .artifacts import load_candidate, write_candidate
from .data import Dataset
from .models import AdvisoryMLP, metrics


def verify_source(expected_revision: str) -> dict:
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise ValueError("full_source_revision_required")
    package = Path(__file__).resolve().parent
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(package), *args], check=True,
                                capture_output=True, text=True, timeout=15)
        return result.stdout.strip()
    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    actual = git("rev-parse", "HEAD")
    if actual != expected_revision or git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source_revision_mismatch_or_dirty_checkout")
    sources = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            rel = path.relative_to(root).as_posix()
            git("ls-files", "--error-unmatch", f":(top,literal){rel}")
            sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"verification": "GIT_CLEAN", "revision": actual, "source_files_sha256": sources,
            "upstream_admission_verified": False,
            "note": "Clean local Git identity only; protected-main admission must be checked by existing Forge release controls."}


def tensors(dataset: Dataset, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows = dataset.select(split)
    return (torch.tensor([[r.features[n] for n in dataset.track.features] for r in rows], dtype=torch.float32),
            torch.tensor([float(r.label) for r in rows], dtype=torch.float32))


def fit(dataset: Dataset, *, epochs: int = 30, seed: int = 20260913, batch_size: int = 32) -> tuple[AdvisoryMLP, dict]:
    if type(epochs) is not int or not 1 <= epochs <= 500 or type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("training_budget_out_of_bounds")
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError("invalid_seed")
    # Preserve surrounding process RNG state; tests and callers are not reseeded globally.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AdvisoryMLP(dataset.track.slug)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x, y = tensors(dataset, "train")
        gen = torch.Generator().manual_seed(seed)
        model.train()
        for _ in range(epochs):
            order = torch.randperm(len(x), generator=gen)
            for indices in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[indices]), y[indices])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite_training_loss")
                loss.backward()
                optimizer.step()
        model.eval()
        vx, vy = tensors(dataset, "validation")
        with torch.inference_mode():
            observed = metrics(model(vx), vy)
    # No test-set metric is calculated during fitting or model selection.
    return model, observed


def train_candidate(dataset_path: Path, track: str, output: Path, revision: str,
                    epochs: int, seed: int, batch_size: int) -> dict:
    source = verify_source(revision)
    dataset = Dataset.read(dataset_path, track)
    if output.exists():
        raise ValueError("output_must_not_exist")
    model, validation = fit(dataset, epochs=epochs, seed=seed, batch_size=batch_size)
    if verify_source(revision) != source:
        raise ValueError("source_changed_during_training")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "safetensors", "pydantic")}
    return write_candidate(output, model, source=source, dataset=dataset.summary(), validation=validation,
                           recipe={"epochs": epochs, "seed": seed, "batch_size": batch_size,
                                   "optimizer": "Adam", "learning_rate": 0.01, "device": "cpu",
                                   "dependencies_observed": versions})


def evaluate_test(artifact: Path, dataset_path: Path) -> dict:
    model, config, digest = load_candidate(artifact)
    dataset = Dataset.read(dataset_path, model.track.slug)
    if dataset.sha256 != config["dataset"]["dataset_sha256"]:
        raise ValueError("test_dataset_not_the_training_bound_dataset")
    x, y = tensors(dataset, "test")
    with torch.inference_mode():
        result = metrics(model(x), y)
    return {"schema": "szl.model-lab.test-evaluation/v1", "artifact_manifest_sha256": digest,
            "dataset_sha256": dataset.sha256, "split": "test", "metrics": result,
            "publication_eligible": False, "signed": False,
            "boundary": "One explicit test observation. Repeated inspection invalidates held-out selection claims."}
