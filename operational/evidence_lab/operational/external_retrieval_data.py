"""Pinned, bounded evaluation data preparation; no model calls or admission changes.

SQuAD negatives are valid only for their original paragraph.  The shared index is
deliberately all validation contexts, so those labels are NOT global negatives.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import unicodedata


SEED = "szl-retrieval-v3"
LICENSE = "CC-BY-SA-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
PINS = {
    "squad_v2": {
        "dataset_id": "rajpurkar/squad_v2",
        "revision": "3ffb306f725f7d2ce8394bc1873b24868140c412",
        "config": "squad_v2", "split": "validation",
        "path": "assets/squad-v2/squad_v2/validation-00000-of-00001.parquet",
        "sha256": "0560174ab095c5ac0a8c8dc8da05f1625453c45a77e4ce9cabc6947ddfdd24cb",
        "size_bytes": 1350511,
        "publisher_url": "https://rajpurkar.github.io/SQuAD-explorer/",
        "paper_url": "https://aclanthology.org/P18-2124/",
        "attribution": "Rajpurkar, Jia, Liang (2018), SQuAD 2.0; original SQuAD: Rajpurkar, Zhang, Lopyrev, Liang (2016); Wikipedia contributors.",
        "license": LICENSE, "license_url": LICENSE_URL,
    },
    "hotpot_qa": {
        "dataset_id": "hotpotqa/hotpot_qa",
        "revision": "1908d6afbbead072334abe2965f91bd2709910ab",
        "config": "distractor", "split": "validation",
        "path": "assets/hotpot-qa/distractor/validation-00000-of-00001.parquet",
        "sha256": "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
        "size_bytes": 27452575,
        "publisher_url": "https://hotpotqa.github.io/",
        "paper_url": "https://aclanthology.org/D18-1259/",
        "attribution": "Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning (2018), HotpotQA; Wikipedia contributors.",
        "license": LICENSE, "license_url": LICENSE_URL,
    },
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _space(text: str) -> str:
    return " ".join(text.split())


def _fold(text: str) -> str:
    return _space(unicodedata.normalize("NFKC", text)).casefold()


def _title_key(text: str) -> str:
    return _fold(text.replace("_", " "))


def _order(kind: str, identity: str) -> tuple[str, str]:
    return _sha(f"{SEED}\0{kind}\0{identity}"), identity


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _question_ok(value) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 512


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _citation(dataset: str, source_ids: list[str]) -> dict:
    pin = PINS[dataset]
    return {key: pin[key] for key in (
        "dataset_id", "revision", "config", "split", "publisher_url",
        "paper_url", "attribution", "license", "license_url",
    )} | {"source_ids": sorted(source_ids)}


def _answer_texts(row: dict) -> list[str]:
    """Validate human answer offsets against the untouched upstream paragraph."""
    answers = row.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("SQuAD answers must be a mapping")
    texts, starts = answers.get("text"), answers.get("answer_start")
    if not isinstance(texts, list) or not isinstance(starts, list) or len(texts) != len(starts):
        raise ValueError("SQuAD answer texts and offsets must be aligned lists")
    for answer, start in zip(texts, starts):
        _text(answer, "answer text")
        if type(start) is not int or start < 0 or row["context"][start:start + len(answer)] != answer:
            raise ValueError(f"SQuAD answer span mismatch for {row['id']}")
    return sorted(set(texts))


def _squad_parts(rows: list[dict], calibration_per_label: int, evaluation_per_label: int):
    parents: dict[str, str] = {}
    records, contexts, seen_ids = [], {}, set()

    def find(title):
        parents.setdefault(title, title)
        while parents[title] != title:
            parents[title] = parents[parents[title]]
            title = parents[title]
        return title

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("SQuAD row must be a mapping")
        row_id = _text(row.get("id"), "SQuAD id")
        if row_id in seen_ids:
            raise ValueError("duplicate SQuAD id")
        seen_ids.add(row_id)
        title = _text(row.get("title"), "SQuAD title")
        original = _text(row.get("context"), "SQuAD context")
        title_key, normalized = _title_key(title), _space(original)
        find(title_key)
        doc_id = "squad-context-" + _sha(normalized)
        context = contexts.setdefault(doc_id, {
            "texts": set(), "titles": set(), "title_keys": set(), "source_ids": set(),
        })
        for other in context["title_keys"]:
            union(title_key, other)
        context["texts"].add(original)
        context["titles"].add(title)
        context["title_keys"].add(title_key)
        context["source_ids"].add(row_id)
        records.append({"id": row_id, "question": row.get("question"),
                        "context_id": doc_id, "answers": _answer_texts(row),
                        "title_key": title_key})

    components = defaultdict(list)
    for title in sorted(parents):
        components[find(title)].append(title)
    group_titles = {"squad-group-" + _sha("\0".join(titles)): titles
                    for titles in components.values()}
    group_for_title = {title: group for group, titles in group_titles.items() for title in titles}
    groups = sorted(group_titles, key=lambda group: _order("component", group))
    if len(groups) < 2:
        raise ValueError("source-disjoint evaluation requires at least two article components")
    boundary = math.ceil(len(groups) / 3)
    split_groups = {"calibration": groups[:boundary], "evaluation": groups[boundary:]}
    assignments = {group: split for split, items in split_groups.items() for group in items}
    question_groups = defaultdict(set)
    for record in records:
        record["group_id"] = group_for_title[record.pop("title_key")]
        if _question_ok(record["question"]):
            question_groups[_fold(record["question"])].add(record["group_id"])

    dropped = Counter()
    eligible = {split: {True: [], False: []} for split in split_groups}
    for record in records:
        if not _question_ok(record["question"]):
            dropped["empty_nonstring_or_over_512_character_question"] += 1
            continue
        if len(question_groups[_fold(record["question"])]) > 1:
            dropped["duplicate_question_text_crossing_groups"] += 1
            continue
        record["answerable"] = bool(record["answers"])
        eligible[assignments[record["group_id"]]][record["answerable"]].append(record)

    selected = {}
    for split, quota in (("calibration", calibration_per_label), ("evaluation", evaluation_per_label)):
        selected[split] = []
        for label in (True, False):
            candidates = sorted(eligible[split][label], key=lambda row: _order("query", row["id"]))
            if len(candidates) < quota:
                raise ValueError(f"{split} answerable={label}: need {quota}, have {len(candidates)}; no quota fallback")
            selected[split].extend(candidates[:quota])
        selected[split].sort(key=lambda row: _order("query", row["id"]))

    documents = []
    for doc_id, context in sorted(contexts.items()):
        titles = sorted(context["titles"])
        citation = _citation("squad_v2", sorted(context["source_ids"]))
        citation["normalized_context_sha256"] = doc_id.removeprefix("squad-context-")
        citation["original_context_sha256s"] = sorted(_sha(text) for text in context["texts"])
        documents.append({"id": doc_id, "title": titles[0], "titles": titles,
                          "text": min(context["texts"]), "citation": citation})
    group_metadata = {
        "seed": SEED, "group_titles": group_titles,
        "calibration_groups": split_groups["calibration"],
        "evaluation_groups": split_groups["evaluation"],
        "assignment_rule": "connected article components sharing whitespace-normalized context; seeded component SHA order; first ceil(n/3) calibration",
        "eligible_counts": {split: {"answerable": len(labels[True]), "unsupported": len(labels[False])}
                            for split, labels in eligible.items()},
        "excluded_queries": dict(sorted(dropped.items())),
    }
    return documents, selected, group_metadata


def _hotpot_case(row: dict) -> dict:
    context, support = row.get("context"), row.get("supporting_facts")
    if not isinstance(context, dict) or not isinstance(support, dict):
        raise ValueError("Hotpot context and supporting_facts must be mappings")
    titles, sentences = context.get("title"), context.get("sentences")
    support_titles, sent_ids = support.get("title"), support.get("sent_id")
    if not isinstance(titles, list) or not isinstance(sentences, list) or not titles or len(titles) != len(sentences):
        raise ValueError("Hotpot context titles and sentence lists must align")
    if not isinstance(support_titles, list) or not isinstance(sent_ids, list) or not support_titles or len(support_titles) != len(sent_ids):
        raise ValueError("Hotpot supporting facts must be nonempty aligned lists")
    documents, by_title = [], {}
    for title, sentence_list in zip(titles, sentences):
        _text(title, "Hotpot document title")
        if title in by_title or not isinstance(sentence_list, list) or not sentence_list:
            raise ValueError("Hotpot context has duplicate title or empty sentence list")
        if any(not isinstance(sentence, str) for sentence in sentence_list):
            raise ValueError("Hotpot sentence must be a string")
        text = _text("".join(sentence_list), "Hotpot document text")
        doc_id = "hotpot-context-" + _sha(_title_key(title) + "\0" + _space(text))
        document = {"id": doc_id, "title": title, "text": text,
                    "citation": _citation("hotpot_qa", [row["id"]])}
        document["citation"]["supporting_sentence_ids"] = []
        documents.append(document)
        by_title[title] = document, sentence_list
    positives = set()
    for title, sent_id in zip(support_titles, sent_ids):
        if not isinstance(title, str) or title not in by_title:
            raise ValueError("Hotpot supporting title absent from candidate documents")
        document, sentence_list = by_title[title]
        if type(sent_id) is not int or not 0 <= sent_id < len(sentence_list) or not sentence_list[sent_id].strip():
            raise ValueError("Hotpot supporting sentence index is invalid")
        document["citation"]["supporting_sentence_ids"].append(sent_id)
        positives.add(document["id"])
    for document in documents:
        document["citation"]["supporting_sentence_ids"] = sorted(set(document["citation"]["supporting_sentence_ids"]))
    return {"id": row["id"], "question": row["question"],
            "documents": sorted(documents, key=lambda doc: doc["id"]), "positive_ids": sorted(positives)}


def build_payload(squad_rows: list[dict], hotpot_rows: list[dict], *,
                  calibration_per_label: int = 60, evaluation_per_label: int = 120,
                  hotpot_limit: int = 60, source_receipts: list[dict] | None = None) -> dict:
    """Pure row-to-payload algorithm; custom positive quotas permit small toy tests.

    Raw rows are not authenticated here. Production callers use prepare_payload.
    """
    for name, value in (("calibration_per_label", calibration_per_label),
                        ("evaluation_per_label", evaluation_per_label), ("hotpot_limit", hotpot_limit)):
        _positive_int(value, name)
    documents, selected, groups = _squad_parts(squad_rows, calibration_per_label, evaluation_per_label)
    eligible_hotpot, seen = [], set()
    for row in hotpot_rows:
        if not isinstance(row, dict):
            raise ValueError("Hotpot row must be a mapping")
        row_id = _text(row.get("id"), "Hotpot id")
        if row_id in seen:
            raise ValueError("duplicate Hotpot id")
        seen.add(row_id)
        if _question_ok(row.get("question")):
            eligible_hotpot.append(row)
    if len(eligible_hotpot) < hotpot_limit:
        raise ValueError(f"Hotpot needs {hotpot_limit} valid questions, have {len(eligible_hotpot)}; no quota fallback")
    hotpot = [_hotpot_case(row) for row in sorted(eligible_hotpot, key=lambda row: _order("hotpot", row["id"]))[:hotpot_limit]]
    payload = {
        "schema_version": 1, "documents": documents,
        "calibration": selected["calibration"], "evaluation": selected["evaluation"],
        "hotpot": hotpot,
        "metadata": {
            "dataset_pins": deepcopy(PINS), "source_files": deepcopy(source_receipts or []),
            "source_verification": "PINNED_FILE_SHA256_VERIFIED" if source_receipts else "UNVERIFIED_ROWS_FOR_ALGORITHM_TESTS",
            "selection": {"seed": SEED, "calibration_per_label": calibration_per_label,
                          "evaluation_per_label": evaluation_per_label, "hotpot_limit": hotpot_limit},
            "squad_source_groups": groups,
            "counts": {"squad_source_rows": len(squad_rows), "documents": len(documents),
                       "calibration": len(selected["calibration"]), "evaluation": len(selected["evaluation"]),
                       "hotpot_source_rows": len(hotpot_rows), "hotpot": len(hotpot)},
            "human_label_derivation": {
                "squad": "Positive paragraph qrels derived from human answer spans, checked against original context; not exhaustive human corpus relevance judgments.",
                "squad_unsupported": "Human unanswerable labels refer only to the exact original paragraph, with known annotation noise; not global corpus negatives.",
                "hotpot": "Human supporting-sentence annotations derive positive document IDs; evaluation-only candidate pools; no unsupported labels or calibration.",
            },
            "scope_caveats": [
                "The shared SQuAD index intentionally includes ALL validation contexts, including calibration documents as distractors.",
                "Selected SQuAD query source groups are disjoint; the corpus itself is not split or source-disjoint.",
                "SQuAD original-paragraph unsupported labels cannot support global-corpus false-acceptance claims.",
                "Hotpot uses its provided distractor candidate pools; no all-source independence across Hotpot cases or across datasets is claimed.",
                "Human annotations may be incomplete or noisy. Derived qrels are not exhaustive relevance judgments.",
                "Source-group separation does not establish absence of pretraining, fine-tuning, distillation, or previous model-selection contamination.",
                "This is local evaluation data, not official hidden-test, runtime, deployment, training, or publication evidence.",
            ],
            "license": LICENSE, "license_url": LICENSE_URL,
            "license_basis": "Publisher statements independently checked; preserve attribution, notices, source links and modification records; respect ShareAlike for adapted-data redistribution.",
            "dataset_notes_are_legal_certification": False,
            "training_eligible": False, "publication_eligible": False,
            "pretraining_contamination": "UNKNOWN",
        },
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict) -> None:
    """Fail closed on broken schema, labels, quotas, provenance or source splits."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported payload schema")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("payload metadata is required")
    if metadata.get("training_eligible") is not False or metadata.get("publication_eligible") is not False:
        raise ValueError("evaluation data is ineligible for training and publication")
    if metadata.get("pretraining_contamination") != "UNKNOWN" or metadata.get("dataset_notes_are_legal_certification") is not False:
        raise ValueError("unsupported contamination or legal certification claim")
    if metadata.get("dataset_pins") != PINS:
        raise ValueError("dataset pins do not match fixed inputs")
    selection, source_groups = metadata.get("selection", {}), metadata.get("squad_source_groups", {})
    if selection.get("seed") != SEED:
        raise ValueError("selection seed mismatch")
    group_titles = source_groups.get("group_titles", {})
    group_sets = {split: set(source_groups.get(split + "_groups", [])) for split in ("calibration", "evaluation")}
    if not all(group_sets.values()) or group_sets["calibration"] & group_sets["evaluation"]:
        raise ValueError("calibration and evaluation source groups must be nonempty and disjoint")
    if set(group_titles) != group_sets["calibration"] | group_sets["evaluation"]:
        raise ValueError("source group assignments must cover all article components")
    title_group = {}
    for group, titles in group_titles.items():
        if not isinstance(titles, list) or not titles or titles != sorted(set(titles)):
            raise ValueError("invalid source group titles")
        if group != "squad-group-" + _sha("\0".join(titles)):
            raise ValueError("source group ID does not match titles")
        for title in titles:
            if title in title_group:
                raise ValueError("article title belongs to multiple source groups")
            title_group[title] = group
    ordered_groups = sorted(group_titles, key=lambda group: _order("component", group))
    if group_sets["calibration"] != set(ordered_groups[:math.ceil(len(ordered_groups) / 3)]):
        raise ValueError("source group assignment violates fixed component split")
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("nonempty documents list required")
    by_id, doc_group = {}, {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("document must be a mapping")
        doc_id, text = _text(document.get("id"), "document id"), _text(document.get("text"), "document text")
        if doc_id in by_id or doc_id != "squad-context-" + _sha(_space(text)):
            raise ValueError("duplicate or content-mismatched document id")
        titles = document.get("titles")
        if not isinstance(titles, list) or not titles or any(not isinstance(title, str) or not title.strip() for title in titles):
            raise ValueError("document titles must be nonempty strings")
        if titles != sorted(set(titles)) or document.get("title") != titles[0]:
            raise ValueError("document title aliases are not canonical")
        groups = {title_group.get(_title_key(title)) for title in titles}
        if None in groups or len(groups) != 1:
            raise ValueError("document context crosses source groups")
        citation = document.get("citation", {})
        if citation.get("dataset_id") != PINS["squad_v2"]["dataset_id"] or citation.get("revision") != PINS["squad_v2"]["revision"] or citation.get("license") != LICENSE:
            raise ValueError("document citation does not match pinned licensed source")
        by_id[doc_id], doc_group[doc_id] = document, groups.pop()
    if list(by_id) != sorted(by_id):
        raise ValueError("documents must be sorted by stable ID")
    seen_queries, question_splits, context_splits = set(), {}, {}
    for split in ("calibration", "evaluation"):
        rows = payload.get(split)
        quota = _positive_int(selection.get(split + "_per_label"), split + "_per_label")
        if not isinstance(rows, list) or len(rows) != 2 * quota:
            raise ValueError(f"{split} query quota mismatch")
        counts = Counter()
        for row in rows:
            if not isinstance(row, dict) or not _question_ok(row.get("question")):
                raise ValueError("invalid selected question")
            row_id = _text(row.get("id"), "query id")
            if row_id in seen_queries:
                raise ValueError("duplicate selected query id")
            seen_queries.add(row_id)
            context_id, group = row.get("context_id"), row.get("group_id")
            if context_id not in by_id or group not in group_sets[split] or doc_group[context_id] != group:
                raise ValueError("query source group/context is missing or assigned to wrong split")
            answers, label = row.get("answers"), row.get("answerable")
            if not isinstance(answers, list) or type(label) is not bool or label != bool(answers):
                raise ValueError("answerability label does not match answers")
            if any(not isinstance(answer, str) or not answer.strip() or _space(answer) not in _space(by_id[context_id]["text"]) for answer in answers):
                raise ValueError("answer not found in original-equivalent context")
            counts[label] += 1
            for value, mapping in ((_fold(row["question"]), question_splits), (context_id, context_splits)):
                if value in mapping and mapping[value] != split:
                    raise ValueError("query text or context leaks across calibration/evaluation")
                mapping[value] = split
        if counts != {True: quota, False: quota}:
            raise ValueError(f"{split} answerability balance mismatch")
    hotpot = payload.get("hotpot")
    if not isinstance(hotpot, list) or len(hotpot) != _positive_int(selection.get("hotpot_limit"), "hotpot_limit"):
        raise ValueError("Hotpot quota mismatch")
    seen_hotpot = set()
    for case in hotpot:
        if not isinstance(case, dict) or not _question_ok(case.get("question")):
            raise ValueError("invalid Hotpot case")
        case_id = _text(case.get("id"), "Hotpot case id")
        if case_id in seen_hotpot:
            raise ValueError("duplicate Hotpot case id")
        seen_hotpot.add(case_id)
        if "answerable" in case or "unsupported" in case or "calibration" in case:
            raise ValueError("Hotpot cases carry no unsupported labels or calibration")
        docs = case.get("documents")
        if not isinstance(docs, list) or not docs:
            raise ValueError("Hotpot candidate documents required")
        doc_ids = []
        for doc in docs:
            if not isinstance(doc, dict):
                raise ValueError("Hotpot document must be a mapping")
            title, text = _text(doc.get("title"), "Hotpot title"), _text(doc.get("text"), "Hotpot text")
            doc_id = "hotpot-context-" + _sha(_title_key(title) + "\0" + _space(text))
            if doc.get("id") != doc_id:
                raise ValueError("Hotpot document content ID mismatch")
            citation = doc.get("citation", {})
            if citation.get("dataset_id") != PINS["hotpot_qa"]["dataset_id"] or citation.get("revision") != PINS["hotpot_qa"]["revision"] or citation.get("license") != LICENSE:
                raise ValueError("Hotpot citation mismatch")
            doc_ids.append(doc_id)
        positives = case.get("positive_ids")
        if len(set(doc_ids)) != len(doc_ids) or not isinstance(positives, list) or not positives or any(not isinstance(item, str) for item in positives):
            raise ValueError("Hotpot document/positive IDs must be nonempty and unique")
        if len(set(positives)) != len(positives) or not set(positives) <= set(doc_ids):
            raise ValueError("Hotpot positive document is absent or duplicated")
    for name in ("documents", "calibration", "evaluation", "hotpot"):
        if metadata.get("counts", {}).get(name) != len(payload[name]):
            raise ValueError("payload counts do not match contents")


def _verified_file_bytes(path: Path, pin: dict) -> tuple[bytes, dict]:
    with path.open("rb") as handle:
        data = handle.read(pin["size_bytes"] + 1)
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != pin["size_bytes"] or digest != pin["sha256"]:
        raise ValueError(f"pinned validation file size/SHA-256 mismatch: {pin['path']}")
    return data, {"dataset_id": pin["dataset_id"], "path": pin["path"],
                  "sha256": digest, "size_bytes": len(data), "verified": True}


def prepare_payload(labroot: Path) -> dict:
    """Verify both fixed files before importing pyarrow or parsing either payload.

    Parse the already-verified bytes, avoiding a file-replacement race between
    hashing and parsing. This performs no download, writes, or model calls.
    """
    verified = [_verified_file_bytes(Path(labroot) / pin["path"], pin) for pin in PINS.values()]
    import pyarrow as pa
    import pyarrow.parquet as pq

    squad_rows, hotpot_rows = [pq.read_table(pa.BufferReader(data)).to_pylist() for data, _ in verified]
    return build_payload(squad_rows, hotpot_rows, source_receipts=[receipt for _, receipt in verified])
