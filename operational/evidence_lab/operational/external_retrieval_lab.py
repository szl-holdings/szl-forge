"""Pinned, local, EVAL-ONLY retrieval and extractive QA; no training/publication.

Run --help. Companion data/core/reader modules are required. Generated datasets
retain their own CC BY-SA license, separate from this repository's code license.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from external_retrieval_core import (BM25, stable_order, rrf_fusion,
    retrieval_metrics, qa_exact_f1, answerability_metrics, choose_threshold)
from external_retrieval_data import prepare_payload, validate_payload

SCOPE = "EXTERNAL_BENCHMARK_EVAL_ONLY"
CONFIG = {
    "encoder": "Qwen/Qwen3-Embedding-0.6B",
    "encoder_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "reader": "deepset/roberta-base-squad2",
    "reader_revision": "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4",
    "encoder_max_tokens": 512, "encoder_batch": 8,
    "encoder_dtype_cuda": "bfloat16", "reader_dtype": "float32",
    "reader_max_tokens": 384, "reader_stride": 128, "reader_max_windows": 8,
    "reader_max_answer_tokens": 30, "reader_n_best": 20,
    "query_instruction": "Given a web search query, retrieve relevant passages that answer the query",
    "rrf_k": 60, "retrieved_reader_documents": 3,
    "calibration_max_false_answer_rate": 0.05,
    "decision_rule": "answer iff span_minus_null > threshold",
    "training_performed": False, "publication_eligible": False,
}
QWEN_FILES = {
    "config.json": "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    "tokenizer_config.json": "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
READER_FILES = {
    "model.safetensors": "ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7",
    "config.json": "64fa58495a722d57609c22f199824bfe98c19be068136a70c268214a08cb8060",
    "vocab.json": "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "tokenizer_config.json": "7a33226d4265e3989cc6341666af179d0cc710136f4059aae0dd8c0797cba556",
    "special_tokens_map.json": "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",
    "README.md": "04834cd9007b2cdd1b1bdd501b571579577331e467be6e24e47d40f060558a29",
}
DATA_FILES = {
    "squad-v2/squad_v2/validation-00000-of-00001.parquet": "0560174ab095c5ac0a8c8dc8da05f1625453c45a77e4ce9cabc6947ddfdd24cb",
    "hotpot-qa/distractor/validation-00000-of-00001.parquet": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def save_new(path, value):
    serialized = json_bytes(value)
    with Path(path).open("xb") as handle:
        handle.write(serialized)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def verify_files(root, expected):
    receipts = {}
    for name, expected_sha in expected.items():
        path = Path(root) / name
        if not path.is_file() or path.is_symlink() or digest(path) != expected_sha:
            raise ValueError(f"Pinned asset integrity failure: {name}")
        receipts[name] = {"sha256": expected_sha, "bytes": path.stat().st_size}
    return receipts


def code_receipt():
    return {name: digest(HERE / name) for name in (
        "external_retrieval_lab.py", "external_retrieval_core.py",
        "external_retrieval_data.py", "external_retrieval_reader.py")}


def paths(labroot, run_id):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must be a simple 1-64 character identifier")
    return Path(labroot).resolve(), Path(labroot).resolve() / "runs" / run_id


def verify_assets(labroot, encoder_path):
    # Native loaders can prefer optional tokenizer/adapter files. An unpinned
    # extra file must not silently change the runtime selected by these hashes.
    for directory, expected in ((Path(encoder_path), QWEN_FILES),
                                (labroot / "assets" / "reader", READER_FILES)):
        allowed = set(expected) | {"README.md", "LICENSE", "NOTICE", ".cache"}
        if any(path.name not in allowed for path in directory.iterdir()):
            raise ValueError("Unexpected file in pinned model directory")
    return {"encoder": verify_files(encoder_path, QWEN_FILES),
            "reader": verify_files(labroot / "assets" / "reader", READER_FILES),
            "datasets": verify_files(labroot / "assets", DATA_FILES)}


def prepare(labroot, run_id, encoder_path, device):
    labroot, run = paths(labroot, run_id)
    assets = verify_assets(labroot, encoder_path)
    payload = prepare_payload(labroot)
    validate_payload(payload)
    run.mkdir(parents=True, exist_ok=False)
    save_new(run / "payload.json", payload)
    freeze = {"schema_version": 1, "scope": SCOPE, "created_at": utc(),
              "config": CONFIG, "device": device,
              "encoder_path": str(Path(encoder_path).resolve()),
              "assets": assets, "code": code_receipt(),
              "payload_sha256": digest(run / "payload.json"),
              "calibration_label_scope": "original_given_context_only",
              "corpus_global_unsupported_judgments": "UNAVAILABLE",
              "contamination": "UNKNOWN", "publication_eligible": False}
    save_new(run / "freeze.json", freeze)
    return {"run": str(run), "freeze_sha256": digest(run / "freeze.json"),
            "documents": len(payload["documents"]),
            "calibration": len(payload["calibration"]),
            "evaluation": len(payload["evaluation"]), "hotpot": len(payload["hotpot"])}


def load_frozen(labroot, run_id):
    labroot, run = paths(labroot, run_id)
    freeze = load_json(run / "freeze.json")
    if freeze["config"] != CONFIG or freeze["code"] != code_receipt():
        raise ValueError("Runtime/config changed since freeze; create a new run")
    if digest(run / "payload.json") != freeze["payload_sha256"]:
        raise ValueError("Frozen dataset was modified")
    if verify_assets(labroot, freeze["encoder_path"]) != freeze["assets"]:
        raise ValueError("Asset metadata changed")
    payload = load_json(run / "payload.json")
    validate_payload(payload)
    return labroot, run, freeze, payload


def last_token_pool(hidden, attention_mask):
    """Correct for either padding side; reject entirely padded rows."""
    import torch
    if (hidden.ndim != 3 or any(dimension <= 0 for dimension in hidden.shape)
            or attention_mask.shape != hidden.shape[:2]):
        raise ValueError("Invalid pooling dimensions")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("Invalid attention mask")
    positions = torch.arange(attention_mask.shape[1], device=hidden.device)
    last = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(1).values
    if torch.any(last < 0):
        raise ValueError("All-padding embedding")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def offline():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")


class Encoder:
    def __init__(self, path, device):
        offline()
        import torch
        from transformers import AutoTokenizer, AutoModel
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent device change")
        self.device, self.torch = device, torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(device).eval()
        self.truncated = 0
        self.texts = 0

    def encode(self, texts, *, query=False):
        import numpy as np
        chunks = []
        for offset in range(0, len(texts), CONFIG["encoder_batch"]):
            batch = texts[offset:offset + CONFIG["encoder_batch"]]
            if query:
                batch = [f"Instruct: {CONFIG['query_instruction']}\nQuery: {text}" for text in batch]
            full = self.tokenizer(batch, truncation=False, verbose=False)["input_ids"]
            self.truncated += sum(len(ids) > CONFIG["encoder_max_tokens"] for ids in full)
            self.texts += len(batch)
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                max_length=CONFIG["encoder_max_tokens"], return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                output = self.model(**encoded, use_cache=False).last_hidden_state
                pooled = last_token_pool(output, encoded["attention_mask"]).float()
                norm = pooled.norm(dim=1, keepdim=True)
                if not self.torch.isfinite(pooled).all() or (norm <= 0).any():
                    raise ValueError("Invalid embedding")
                chunks.append((pooled / norm).cpu().numpy())
        if not chunks:
            raise ValueError("No texts to encode")
        return np.concatenate(chunks).astype(np.float32, copy=False)


def reader_records(reader, cases, doc_by_id, *, progress):
    records = []
    for i, case in enumerate(cases):
        pred = reader.predict(case["question"], doc_by_id[case["context_id"]]["text"])
        records.append({**pred, "id": case["id"], "answerable": case["answerable"],
                        "gold": case["answers"]})
        if (i + 1) % 30 == 0:
            print(f"{progress}: {i + 1}/{len(cases)}", flush=True)
    return records


def rankings(question, query_vector, documents, vectors, lexical):
    bm25 = lexical.scores(question)
    dense = (vectors @ query_vector).tolist()
    return {"bm25": bm25, "qwen": dense, "hybrid_rrf": rrf_fusion([bm25, dense])}


def evidence_answer(pred, doc, threshold):
    if (isinstance(threshold, bool) or not math.isfinite(threshold) or
            type(pred["start"]) is not int or type(pred["end"]) is not int or
            not 0 <= pred["start"] < pred["end"] <= len(doc["text"]) or
            isinstance(pred["margin"], bool) or not math.isfinite(pred["margin"]) or
            doc["text"][pred["start"]:pred["end"]] != pred["prediction"]):
        raise ValueError("Invalid answer/source binding")
    selected = bool(pred["prediction"].strip()) and pred["margin"] > threshold
    return {"status": "ANSWER" if selected else "ABSTAIN", "answer": pred["prediction"] if selected else None,
            "margin": pred["margin"], "threshold": threshold,
            "confidence_probability": None,
            "evidence": {"document_id": doc["id"], "title": doc["title"],
                "citation": doc["citation"], "context_sha256": hashlib.sha256(doc["text"].encode()).hexdigest(),
                "start": pred["start"], "end": pred["end"]} if selected else None}


def evaluate(labroot, run_id):
    labroot, run, freeze, payload = load_frozen(labroot, run_id)
    save_new(run / "attempt.json", {"started_at": utc(), "freeze_sha256": digest(run / "freeze.json")})
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        import transformers
        from external_retrieval_reader import ExtractiveReader
        encoder = Encoder(freeze["encoder_path"], freeze["device"])
        reader = ExtractiveReader(labroot / "assets" / "reader", device=freeze["device"])
        documents = payload["documents"]
        by_id = {d["id"]: d for d in documents}
        print(f"Models loaded; calibrating {len(payload['calibration'])} given-context questions", flush=True)
        calibration_rows = reader_records(reader, payload["calibration"], by_id, progress="calibration")
        calibration = {**choose_threshold(calibration_rows, CONFIG["calibration_max_false_answer_rate"]),
            "created_at": utc(), "scope": "original_given_context_only", "records": calibration_rows,
            "freeze_sha256": digest(run / "freeze.json")}
        save_new(run / "calibration.json", calibration)
        calibration_hash = digest(run / "calibration.json")
        threshold = calibration["threshold"]
        print(f"Calibration frozen, threshold={threshold:.6f}; evaluation starts now", flush=True)
        evaluation_rows = reader_records(reader, payload["evaluation"], by_id, progress="given-context evaluation")
        print(f"Encoding {len(documents)} corpus passages", flush=True)
        vectors = encoder.encode([d["text"] for d in documents])
        with (run / "documents.npy").open("xb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        positives = [q for q in payload["evaluation"] if q["answerable"]]
        query_vectors = encoder.encode([q["question"] for q in positives], query=True)
        lexical = BM25([d["text"] for d in documents])
        positions = {d["id"]: i for i, d in enumerate(documents)}
        rows = {name: [] for name in ("bm25", "qwen", "hybrid_rrf")}
        retrieval_records, end_to_end = [], []
        for i, case in enumerate(positives):
            scored = rankings(case["question"], query_vectors[i], documents, vectors, lexical)
            record = {"id": case["id"], "positive_id": case["context_id"], "ranks": {}}
            for name, scores in scored.items():
                rows[name].append(scores)
                order = stable_order(scores)
                record["ranks"][name] = order.index(positions[case["context_id"]]) + 1
            candidates = []
            for position in stable_order(scored["hybrid_rrf"])[:CONFIG["retrieved_reader_documents"]]:
                doc = documents[position]
                candidates.append((reader.predict(case["question"], doc["text"]), doc))
            pred, doc = max(candidates, key=lambda pair: pair[0]["margin"])
            answer = evidence_answer(pred, doc, threshold)
            quality = qa_exact_f1(answer["answer"] or "", case["answers"])
            end_to_end.append({"id": case["id"], **answer, **quality})
            retrieval_records.append(record)
            if (i + 1) % 30 == 0:
                print(f"Retrieval + top-three reader: {i + 1}/{len(positives)}", flush=True)
        relevant = [[positions[q["context_id"]]] for q in positives]
        retrieval = {name: retrieval_metrics(scores, relevant) for name, scores in rows.items()}
        hotpot_rows = {name: [] for name in rows}
        hotpot_qrels, hotpot_records = [], []
        # Shared encoding cache saves repeated paragraphs without changing candidate pools.
        hotpot_texts = sorted({d["text"] for q in payload["hotpot"] for d in q["documents"]})
        hotpot_vectors = encoder.encode(hotpot_texts)
        hotpot_map = {text: vec for text, vec in zip(hotpot_texts, hotpot_vectors)}
        hotpot_queries = encoder.encode([q["question"] for q in payload["hotpot"]], query=True)
        for i, case in enumerate(payload["hotpot"]):
            docs = case["documents"]
            local_vectors = np.stack([hotpot_map[d["text"]] for d in docs])
            scored = rankings(case["question"], hotpot_queries[i], docs, local_vectors, BM25([d["text"] for d in docs]))
            positive_positions = [j for j, d in enumerate(docs) if d["id"] in case["positive_ids"]]
            hotpot_qrels.append(positive_positions)
            ranks = {}
            for name, scores in scored.items():
                hotpot_rows[name].append(scores)
                order = stable_order(scores)
                ranks[name] = [order.index(j) + 1 for j in positive_positions]
            hotpot_records.append({"id": case["id"], "support_ranks": ranks})
        hotpot_metrics = {name: {**retrieval_metrics(scores, hotpot_qrels),
            "all_support_at_2": sum(max(r["support_ranks"][name]) <= 2 for r in hotpot_records) / len(hotpot_records),
            "all_support_at_5": sum(max(r["support_ranks"][name]) <= 5 for r in hotpot_records) / len(hotpot_records)}
            for name, scores in hotpot_rows.items()}
        # Revalidate source and payload before writing the completed result.
        load_frozen(labroot, run_id)
        if digest(run / "calibration.json") != calibration_hash:
            raise ValueError("Calibration changed during evaluation")
        result = {"status": "LOCAL_EVALUATION_COMPLETED_NOT_PRODUCTION", "scope": SCOPE,
            "completed_at": utc(), "elapsed_seconds": time.monotonic() - started,
            "freeze_sha256": digest(run / "freeze.json"), "calibration_sha256": calibration_hash,
            "index_sha256": digest(run / "documents.npy"),
            "runtime": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                "device": freeze["device"], "gpu": torch.cuda.get_device_name(0) if freeze["device"] == "cuda" else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if freeze["device"] == "cuda" else None,
                "encoder_texts": encoder.texts, "encoder_truncated_texts": encoder.truncated},
            "dataset_metadata": payload["metadata"],
            "squad_retrieval": retrieval, "hotpot_support_retrieval": hotpot_metrics,
            "given_context_answerability": answerability_metrics(evaluation_rows, threshold),
            "retrieved_context_qa_answerable_only": {"queries": len(end_to_end),
                "exact_match": sum(r["exact_match"] for r in end_to_end) / len(end_to_end),
                "f1": sum(r["f1"] for r in end_to_end) / len(end_to_end),
                "coverage": sum(r["status"] == "ANSWER" for r in end_to_end) / len(end_to_end)},
            "limitations": ["Public development sets; contamination UNKNOWN, not hidden benchmark results",
                "SQuAD negative labels apply only to the original paragraph",
                "Retrieved-document selection changes distribution; given-context false-answer rate does not transfer",
                "Supporting-fact positives are not exhaustive corpus-wide relevance judgments",
                "Hotpot uses native ten-document distractor pools, not full Wikipedia search",
                "No training, provider publication, independent witnessing or production authorization"],
            "training_performed": False, "publication_eligible": False,
            "records": {"given_context": evaluation_rows, "retrieval": retrieval_records,
                        "hotpot": hotpot_records, "retrieved_context_qa": end_to_end}}
        save_new(run / "result.json", result)
        return {key: value for key, value in result.items() if key not in ("records", "dataset_metadata")}
    except Exception as exc:
        save_new(run / "failed.json", {"failed_at": utc(), "type": type(exc).__name__,
                                      "elapsed_seconds": time.monotonic() - started})
        raise


def validate_question(question):
    if not isinstance(question, str) or not question.strip() or len(question) > 512:
        raise ValueError("Question must contain 1-512 characters")
    return question.strip()


class LocalService:
    def __init__(self, labroot, run_id):
        import numpy as np
        from external_retrieval_reader import ExtractiveReader
        labroot, run, freeze, self.payload = load_frozen(labroot, run_id)
        result = load_json(run / "result.json")
        if (result["status"] != "LOCAL_EVALUATION_COMPLETED_NOT_PRODUCTION" or
                result["freeze_sha256"] != digest(run / "freeze.json") or
                result["index_sha256"] != digest(run / "documents.npy") or
                result["calibration_sha256"] != digest(run / "calibration.json")):
            raise ValueError("Completed run artifact binding failed")
        calibration = load_json(run / "calibration.json")
        if calibration["freeze_sha256"] != result["freeze_sha256"]:
            raise ValueError("Calibration binding failed")
        self.threshold = calibration["threshold"]
        if isinstance(self.threshold, bool) or not math.isfinite(self.threshold):
            raise ValueError("Invalid threshold")
        self.documents = self.payload["documents"]
        self.vectors = np.load(run / "documents.npy", allow_pickle=False)
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.documents) or not np.isfinite(self.vectors).all():
            raise ValueError("Invalid frozen index")
        self.encoder = Encoder(freeze["encoder_path"], freeze["device"])
        self.reader = ExtractiveReader(labroot / "assets" / "reader", device=freeze["device"])
        self.lexical = BM25([d["text"] for d in self.documents])
        self.run_id, self.freeze_sha = run_id, result["freeze_sha256"]
        self.lock = threading.Lock()

    def search(self, question, k=5):
        question = validate_question(question)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
            raise ValueError("k must be an integer in 1..10")
        query = self.encoder.encode([question], query=True)[0]
        scores = rankings(question, query, self.documents, self.vectors, self.lexical)["hybrid_rrf"]
        return [{**self.documents[i], "rank": rank + 1, "retrieval_score": scores[i]}
                for rank, i in enumerate(stable_order(scores)[:k])]

    def query(self, question):
        hits = self.search(question, CONFIG["retrieved_reader_documents"])
        candidates = [(self.reader.predict(question, d["text"]), d) for d in hits]
        pred, doc = max(candidates, key=lambda pair: pair[0]["margin"])
        return {**evidence_answer(pred, doc, self.threshold), "passages": hits,
                "scope": "retrieved_passages_only", "global_unanswerability": "UNVERIFIED",
                "false_answer_rate_after_retrieval_selection": "UNMEASURED",
                "run_id": self.run_id, "freeze_sha256": self.freeze_sha}

    def answer_context(self, question, context):
        question = validate_question(question)
        if not isinstance(context, str) or not context.strip() or len(context) > 16000:
            raise ValueError("Context must contain 1-16000 characters")
        doc = {"id": "provided-" + hashlib.sha256(context.encode()).hexdigest(),
               "title": "User-provided context", "citation": "user-provided; rights not reviewed", "text": context}
        pred = self.reader.predict(question, context)
        return {**evidence_answer(pred, doc, self.threshold), "scope": "provided_context_only",
                "calibration_domain": "SQuAD2; transfer to supplied context not established"}


def create_app(service):
    from fastapi import FastAPI, Request, HTTPException
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    app = FastAPI(title="SZL Local Evidence Retrieval", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.get("/health")
    def health():
        return {"status": "LOCAL_READY_NOT_PRODUCTION", "scope": SCOPE,
                "documents": len(service.documents), "run_id": service.run_id,
                "freeze_sha256": service.freeze_sha, "training_performed": False}

    # Runtime Request annotation is installed below because imports are lazy.
    async def invoke(request):
        if request.headers.get("origin") is not None:
            raise HTTPException(status_code=403, detail="Browser cross-origin access disabled")
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise HTTPException(status_code=415, detail="application/json required")
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 65536:
                raise HTTPException(status_code=413, detail="Request body exceeds 64 KiB")
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            allowed = {"/search": {"question", "k"}, "/query": {"question"},
                       "/answer-context": {"question", "context"}}[request.url.path]
            if not set(body) <= allowed or "question" not in body:
                raise ValueError("Unknown or missing fields")
            if not service.lock.acquire(blocking=False):
                raise HTTPException(status_code=429, detail="Local GPU busy; retry later")
            try:
                # Execute blocking inference off the event loop; only one request owns the GPU.
                from starlette.concurrency import run_in_threadpool
                if request.url.path == "/search":
                    return await run_in_threadpool(service.search, body["question"], body.get("k", 5))
                if request.url.path == "/query":
                    return await run_in_threadpool(service.query, body["question"])
                return await run_in_threadpool(service.answer_context, body["question"], body.get("context"))
            finally:
                service.lock.release()
        except (ValueError, TypeError, KeyError, RecursionError):
            raise HTTPException(status_code=422, detail="Invalid or unsupported input") from None
    invoke.__annotations__["request"] = Request
    for route in ("/search", "/query", "/answer-context"):
        app.add_api_route(route, invoke, methods=["POST"])
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "evaluate", "query", "serve"])
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--question")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.encoder_path is None:
            parser.error("prepare requires --encoder-path")
        result = prepare(args.lab_root, args.run_id, args.encoder_path, args.device)
    elif args.command == "evaluate":
        result = evaluate(args.lab_root, args.run_id)
    else:
        if args.command == "query" and args.question is None:
            parser.error("query requires --question")
        service = LocalService(args.lab_root, args.run_id)
        if args.command == "serve":
            import uvicorn
            if not 1024 <= args.port <= 65535:
                parser.error("port must be in 1024..65535")
            uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, workers=1, access_log=False)
            return
        result = service.query(args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
