"""Closed, bytes-only observation of the local MSQ650 candidate; never admission.

No source data, filesystem access, network, trainer or execution hook is included.
Matching pinned bytes identifies one candidate, not the truth or licensing of its
contents. Source derivation, held-out contamination, rights authority, signatures,
runtime qualification, and training permission remain separate, unclosed gates.
The internal row inspector supports tiny SIMULATED tests, not alternate profiles.
"""

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import re
import unicodedata


PROFILE = "szl.msq650-candidate-v1-byte-observation/v1"
BUILDER_SHA256 = "a292cc43271f3bc54153383b52174d81e9d8c37198bca35290923b5cb74498dd"
VERIFIER_SHA256 = "4367cc3defc1d7462e8993a0e243350190f7123c9e4b18907f6f9f06f4f868cb"
ARTIFACT_PINS = (
    ("manifest", 5959, "e10294badec5f4291e5321c96a925d8629896306da24e77e5d406b3614e6389c"),
    ("train", 952356, "b356cf22656e51544bfd98d39f1dc6895e028f2e58e029b38aece042acbc145b"),
    ("dev", 269347, "e6e428847efe6a89520f80661707856c01f90dd5b21e3a81af3e830d78e5ae31"),
    ("units", 1045, "64ce9e2f533f9df9fc0547dc0a8b70e6c9bfa34e376f82e1530090404a090b80"),
    ("rights", 1995, "cd22060eb0849ca367f08348976f63b8690295e2c3219fe909a9670449cdde36"),
    ("acquisition", 4238, "1e7a9b724b7b9319f5f2c723108c45941c3137ee9cd9dc22bbe19e6bf26f00cc"),
)
_SOURCE_IDS = (
    ("math", "szl-generated-math-v1", "Apache-2.0"),
    ("science", "qudt-3.5.3", "CC-BY-4.0"),
    ("quant", "treasury-debt-2019-as-retrieved-20261023", "Treasury-unrestricted-use"),
)
_HOLDS = (
    "New candidate contract and sealed evaluation commitments",
    "Cross-corpus/held-out semantic overlap review",
    "Dedicated signing policy and external trust provenance",
    "Native dependency/runtime qualification",
    "Qualified Bonsai-preserving coexistence profile",
    "Owned-job cumulative cost/time enforcement",
)
_ROW_FIELDS = frozenset(("id", "domain", "group_id", "split", "source_id", "license", "task", "messages", "provenance"))
_PROVENANCE_FIELDS = frozenset(("source_sha256", "license_evidence_sha256", "source_revision", "retrieved_at", "source_url", "evidence_class"))
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_MAX_BYTES = 2 * 1024 * 1024
_MAX_ROWS = 650
_SYSTEM = (
    "You propose numerical answers, never actions or authoritative receipts. Return only a JSON object "
    "with exactly value (string), unit (string), evidence_class and source_id. Use the supplied structured "
    "facts only. Synthetic math uses SIMULATED; calculations from declared source facts use MODELED. "
    "No tool execution, investment recommendation, measurement or publication claim is authorized."
)
_PROMPT = (
    "Compute the requested value exactly. Use a reduced numerator/denominator for fraction_sum, plain decimal "
    "for other tasks, and exactly two decimal places for debt_sum. Historical records are as retrieved now; "
    "exact original release timestamps are UNKNOWN.\n"
)


class CorpusContractError(ValueError):
    """Bounded, non-secret-bearing rejection code."""


@dataclass(frozen=True, slots=True)
class CorpusObservation:
    """Immutable measurement, intentionally unusable as an authorization bool."""

    candidate_bytes_verified: bool
    structure_verified: bool
    errors: tuple[str, ...]
    row_counts: tuple[tuple[str, int], ...] = ()
    domain_counts: tuple[tuple[str, str, int], ...] = ()
    group_count: int = 0
    profile: str = field(default=PROFILE, init=False)
    evidence_class: str = field(default="MEASURED", init=False)
    state: str = field(default="BLOCKED", init=False)
    artifact_pins: tuple = field(default=ARTIFACT_PINS, init=False)
    builder_sha256: str = field(default=BUILDER_SHA256, init=False)
    verifier_sha256: str = field(default=VERIFIER_SHA256, init=False)
    tool_hashes_observed: bool = field(default=False, init=False)
    source_derivation_replayed: bool = field(default=False, init=False)
    rights_evidence_class: str = field(default="DECLARED", init=False)
    rights_admission: str = field(default="UNAVAILABLE", init=False)
    semantic_overlap_clearance: str = field(default="UNKNOWN", init=False)
    sealed_evaluation: str = field(default="UNAVAILABLE", init=False)
    signing_authority: str = field(default="UNAVAILABLE", init=False)
    runtime_admission: str = field(default="UNAVAILABLE", init=False)
    training_admitted: bool = field(default=False, init=False)
    execution_authority: bool = field(default=False, init=False)
    publication_eligible: bool = field(default=False, init=False)
    heldout_content_opened: bool = field(default=False, init=False)

    def __bool__(self):
        raise TypeError("CorpusObservation is evidence, not an admission decision")


def _require(condition, code):
    if not condition:
        raise CorpusContractError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _no_number(_value):
    raise CorpusContractError("FLOAT_OR_NONFINITE_JSON_NUMBER")


def _strict(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_BYTES, "JSON_BYTES_TYPE_OR_CAP")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_float=_no_number, parse_constant=_no_number)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, CorpusContractError):
            raise
        raise CorpusContractError("INVALID_JSON") from None
    return value


def _fields(value, keys, code):
    _require(type(value) is dict and set(value) == set(keys), code)


def _task(task, domain):
    _require(type(task) is dict and type(task.get("kind")) is str, "TASK_SCHEMA")
    kind = task["kind"]
    fields = {
        "fraction_sum": ("math", ("kind", "a_num", "a_den", "b_num", "b_den")),
        "linear_equation": ("math", ("kind", "a", "b", "c")),
        "unit_conversion": ("science", ("kind", "value", "from_unit", "to_unit")),
        "debt_sum": ("quant", ("kind", "record_date", "src_line_nbr")),
    }
    _require(kind in fields and fields[kind][0] == domain, "TASK_KIND_OR_DOMAIN")
    _fields(task, fields[kind][1], "TASK_FIELDS")
    if domain == "math":
        _require(all(type(v) is int and abs(v) <= 10**9 for k, v in task.items() if k != "kind"), "TASK_INTEGER_REQUIRED")
        _require((kind == "fraction_sum" and task["a_den"] > 0 and task["b_den"] > 0)
                 or (kind == "linear_equation" and task["a"] != 0), "TASK_DENOMINATOR")
    else:
        _require(all(type(v) is str and 0 < len(v) <= 128 for v in task.values()), "TASK_STRING_REQUIRED")
        if domain == "science":
            _require(_DECIMAL.fullmatch(task["value"]) is not None, "TASK_DECIMAL_REQUIRED")
        else:
            _require(re.fullmatch(r"2019-\d{2}-\d{2}", task["record_date"]) is not None
                     and task["src_line_nbr"].isdigit(), "TASK_HISTORICAL_DATE")
    return kind


def _inspect_rows(train, dev):
    """Structural primitive only; its tuple output cannot grant any authority.

    This does not read source snapshots, calculate answers, or assess semantic
    overlap. Production callers must use observe_corpus, whose pins are closed.
    """
    ids, prompts, tasks, groups = set(), set(), set(), {}
    row_counts, domain_counts, task_counts = [], [], Counter()
    source_ids = {d: (s, lic) for d, s, lic in _SOURCE_IDS}
    for split, raw in (("train", train), ("dev", dev)):
        _require(type(raw) is bytes and 0 < len(raw) <= _MAX_BYTES, "JSONL_BYTES_TYPE_OR_CAP")
        _require(raw.endswith(b"\n") and b"\r" not in raw, "JSONL_NEWLINE_PROFILE")
        lines = raw.split(b"\n")[:-1]
        _require(0 < len(lines) <= _MAX_ROWS, "ROW_COUNT_CAP")
        domains = Counter()
        for line in lines:
            _require(0 < len(line) <= 32768, "ROW_BYTES_CAP")
            row = _strict(line)
            _fields(row, _ROW_FIELDS, "ROW_FIELDS")
            _require(_canonical(row) == line, "ROW_NONCANONICAL")
            domain = row["domain"]
            _require(type(domain) is str and domain in source_ids, "ROW_DOMAIN")
            _require(row["split"] == split and (row["source_id"], row["license"]) == source_ids[domain], "ROW_SOURCE_LICENSE_SPLIT")
            row_id, group = row["id"], row["group_id"]
            _require(type(row_id) is str and _HEX.fullmatch(row_id) is not None and row_id not in ids, "ROW_ID_DUPLICATED_OR_INVALID")
            _require(type(group) is str and 0 < len(group) <= 128 and groups.setdefault(group, split) == split, "GROUP_CROSSES_SPLITS")
            ids.add(row_id)
            _require(len(ids) <= _MAX_ROWS, "TOTAL_ROW_COUNT_CAP")
            kind = _task(row["task"], domain)
            task_key = _canonical(row["task"])
            _require(task_key not in tasks, "DUPLICATE_TASK")
            tasks.add(task_key)
            task_counts[kind] += 1
            messages = row["messages"]
            _require(type(messages) is list and len(messages) == 3, "MESSAGE_COUNT")
            for message, role in zip(messages, ("system", "user", "assistant")):
                _fields(message, ("role", "content"), "MESSAGE_FIELDS")
                _require(message["role"] == role and type(message["content"]) is str, "MESSAGE_ROLE_OR_CONTENT")
            _require(messages[0]["content"] == _SYSTEM, "SYSTEM_POLICY")
            prompt = messages[1]["content"]
            _require(prompt.startswith(_PROMPT), "USER_PREFIX")
            query_raw = prompt[len(_PROMPT):].encode("utf-8")
            query = _strict(query_raw)
            _fields(query, ("task", "context", "source_id"), "QUERY_FIELDS")
            _require(query["task"] == row["task"] and type(query["context"]) is dict
                     and query["source_id"] == row["source_id"]
                     and _canonical(query) == query_raw and _sha(query_raw) == row_id, "QUERY_IDENTITY")
            normalized = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
            _require(normalized not in prompts, "DUPLICATE_NORMALIZED_PROMPT")
            prompts.add(normalized)
            answer_raw = messages[2]["content"].encode("utf-8")
            answer = _strict(answer_raw)
            _fields(answer, ("value", "unit", "evidence_class", "source_id"), "ANSWER_FIELDS")
            _require(_canonical(answer) == answer_raw and all(type(v) is str for v in answer.values()), "ANSWER_STRINGS_OR_CANONICAL")
            _require(answer["source_id"] == row["source_id"] and answer["evidence_class"] == ("SIMULATED" if domain == "math" else "MODELED"), "ANSWER_EVIDENCE_CLASS")
            _require(0 < len(answer["value"]) <= 128 and 0 < len(answer["unit"]) <= 128, "ANSWER_LENGTH")
            if kind == "fraction_sum":
                _require(re.fullmatch(r"-?(?:0|[1-9]\d*)/[1-9]\d*", answer["value"]) is not None and answer["unit"] == "rational", "ANSWER_FRACTION_FORMAT")
            else:
                _require(_DECIMAL.fullmatch(answer["value"]) is not None, "ANSWER_DECIMAL_FORMAT")
                if kind == "debt_sum":
                    _require(re.fullmatch(r"(?:0|[1-9]\d*)\.\d{2}", answer["value"]) is not None and answer["unit"] == "USD", "ANSWER_CENTS_FORMAT")
                elif kind == "linear_equation":
                    _require(answer["unit"] == "scalar", "ANSWER_SCALAR_UNIT")
                else:
                    _require(answer["unit"] == row["task"]["to_unit"], "ANSWER_TARGET_UNIT")
            provenance = row["provenance"]
            expected_fields = _PROVENANCE_FIELDS | ({"released_at", "vintage"} if domain == "quant" else set())
            _fields(provenance, expected_fields, "PROVENANCE_FIELDS")
            for key in ("source_sha256", "license_evidence_sha256"):
                _require(type(provenance[key]) is str and _HEX.fullmatch(provenance[key]) is not None, "PROVENANCE_HASH")
            _require(provenance["evidence_class"] == ("SIMULATED" if domain == "math" else "DECLARED"), "PROVENANCE_EVIDENCE_CLASS")
            if domain == "math":
                _require(provenance == {"source_sha256": BUILDER_SHA256, "license_evidence_sha256": BUILDER_SHA256,
                         "source_revision": "original-generator-v1", "retrieved_at": None,
                         "source_url": None, "evidence_class": "SIMULATED"}, "MATH_PROVENANCE")
            if domain == "quant":
                _require(provenance["released_at"] == "UNKNOWN" and provenance["vintage"] == "historical records as retrieved in 2026, not as-known-then", "QUANT_VINTAGE_CLAIM")
                month = row["task"]["record_date"][5:7]
                _require("01" <= month <= "12" and group == "treasury-month-2019-" + month
                         and split == ("train" if month <= "09" else "dev"), "QUANT_TEMPORAL_GROUP")
            domains[domain] += 1
        row_counts.append((split, len(lines)))
        domain_counts.extend((split, d, n) for d, n in sorted(domains.items()))
    return tuple(row_counts), tuple(domain_counts), len(groups), tuple(sorted(task_counts.items()))


def _inspect_metadata(manifest, units, rights, acquisition):
    """Check pinned metadata's invariants; no external authority is inferred."""
    _fields(manifest, ("base_revision", "budget", "builder_sha256", "candidate_state", "counts", "created_at", "evaluation", "files", "holds", "publication_eligible", "schema", "source_acquisition_sha256", "source_entries", "source_registry", "training_admitted"), "MANIFEST_FIELDS")
    _require(manifest["schema"] == "szl.math-science-quant-pilot/v1" and manifest["base_revision"] is None
             and manifest["candidate_state"] == "LOCAL_DATA_CANDIDATE_NOT_TRAINED", "MANIFEST_PROFILE")
    _require(manifest["training_admitted"] is False and manifest["publication_eligible"] is False, "MANIFEST_AUTHORITY")
    _fields(manifest["budget"], ("max_campaign_hours", "max_usd", "training_started"), "BUDGET_FIELDS")
    _require(type(manifest["budget"]["max_campaign_hours"]) is int and manifest["budget"]["max_campaign_hours"] == 50
             and type(manifest["budget"]["max_usd"]) is int and manifest["budget"]["max_usd"] == 2000
             and manifest["budget"]["training_started"] is False, "BUDGET_OR_AUTHORITY")
    _require(manifest["evaluation"] == {"dev_is_not_final": True, "existing_benchmark_decontamination": "NOT_RUN", "final_heldout_created_or_read": False, "semantic_overlap_clearance": "UNKNOWN"}
             and manifest["evaluation"]["dev_is_not_final"] is True
             and manifest["evaluation"]["final_heldout_created_or_read"] is False
             and manifest["holds"] == list(_HOLDS), "EVALUATION_HOLDS")
    pins = {name: (size, digest) for name, size, digest in ARTIFACT_PINS}
    _require(manifest["builder_sha256"] == BUILDER_SHA256 and manifest["source_acquisition_sha256"] == pins["acquisition"][1], "MANIFEST_LINEAGE")
    files = manifest["files"]
    _require(type(files) is list and len(files) == 4, "MANIFEST_FILES")
    seen = set()
    for entry in files:
        _fields(entry, ("path", "bytes", "sha256"), "MANIFEST_FILE_FIELDS")
        name = entry["path"]
        _require(type(name) is str and name in ("train.jsonl", "dev.jsonl", "units.json", "rights.json") and name not in seen, "MANIFEST_PATH_ALIAS")
        seen.add(name)
        _require(type(entry["bytes"]) is int and (entry["bytes"], entry["sha256"]) == pins[name.split(".")[0]], "MANIFEST_FILE_PIN")
    _fields(rights, ("schema", "rights_evidence_class", "publication_eligible", "scope", "raw_source_notice", "sources"), "RIGHTS_FIELDS")
    _require(rights["schema"] == "szl.corpus-rights-record/v1" and rights["rights_evidence_class"] == "DECLARED"
             and rights["publication_eligible"] is False, "RIGHTS_AUTHORITY")
    _fields(rights["sources"], (source for _, source, _ in _SOURCE_IDS), "RIGHTS_SOURCES")
    for _, source, license_name in _SOURCE_IDS:
        _require(type(rights["sources"][source]) is dict and rights["sources"][source].get("license") == license_name, "RIGHTS_LICENSE")
    _fields(units, ("M", "KiloM", "CentiM", "MilliM", "SEC", "MIN", "HR", "GM", "KiloGM"), "UNIT_SET")
    for value in units.values():
        _fields(value, ("multiplier", "dimension"), "UNIT_FIELDS")
        _require(type(value["multiplier"]) is str and _DECIMAL.fullmatch(value["multiplier"]) is not None
                 and type(value["dimension"]) is str and value["dimension"].startswith("http://qudt.org/vocab/dimensionvector/"), "UNIT_TYPES")
    _fields(acquisition, ("schema", "started_at", "finished_at", "status", "training_admitted", "integrity", "collector_sha256", "total_bytes", "chain_tip", "entries"), "ACQUISITION_FIELDS")
    _require(acquisition["schema"] == "szl.corpus-acquisition/v1" and acquisition["training_admitted"] is False
             and acquisition["status"] == "MEASURED" and acquisition["integrity"] == "Unsigned local hash chain, not a signature or independent witness", "ACQUISITION_AUTHORITY")
    _require(type(acquisition["entries"]) is list and len(acquisition["entries"]) == 6, "ACQUISITION_ENTRIES")
    previous, total, entries = None, 0, {}
    for entry in acquisition["entries"]:
        _fields(entry, ("name", "url", "retrieved_at", "content_type", "bytes", "sha256", "git_blob_sha1", "previous_receipt_sha256", "receipt_sha256"), "ACQUISITION_ENTRY_FIELDS")
        _require(type(entry["name"]) is str and entry["name"] not in entries and type(entry["bytes"]) is int and entry["bytes"] > 0, "ACQUISITION_ENTRY_TYPE_OR_ALIAS")
        _require(entry["previous_receipt_sha256"] == previous and entry["receipt_sha256"] == _sha(_canonical({k: v for k, v in entry.items() if k != "receipt_sha256"})), "ACQUISITION_CHAIN")
        entries[entry["name"]] = entry
        total += entry["bytes"]
        previous = entry["receipt_sha256"]
    _require(type(acquisition["total_bytes"]) is int and acquisition["total_bytes"] == total
             and acquisition["chain_tip"] == previous and manifest["source_entries"] == entries, "ACQUISITION_MANIFEST_BINDING")


def observe_corpus(*, manifest: bytes, train: bytes, dev: bytes, units: bytes,
                   rights: bytes, acquisition: bytes) -> CorpusObservation:
    """Observe exactly six immutable buffers under the sole pinned profile.

    A changed or missing artifact is rejected, without caller-supplied replacement
    pins. Builder/verifier hashes are profile metadata, not observed tool bytes.
    Even a complete match returns BLOCKED and all authority flags false.
    """
    buffers = (manifest, train, dev, units, rights, acquisition)
    matched = False
    try:
        _require(all(type(raw) is bytes for raw in buffers), "ARTIFACT_EXACT_BYTES_REQUIRED")
        _require(sum(map(len, buffers)) <= _MAX_BYTES, "ARTIFACT_TOTAL_BYTE_CAP")
        for (name, size, digest), raw in zip(ARTIFACT_PINS, buffers):
            _require(len(raw) == size and _sha(raw) == digest, "ARTIFACT_PIN_MISMATCH:" + name)
        matched = True
        metadata = tuple(_strict(raw) for raw in (manifest, units, rights, acquisition))
        _inspect_metadata(*metadata)
        counts, domains, groups, tasks = _inspect_rows(train, dev)
        _require(counts == (("train", 508), ("dev", 142))
                 and domains == (("train", "math", 192), ("train", "quant", 188), ("train", "science", 128), ("dev", "math", 48), ("dev", "quant", 62), ("dev", "science", 32))
                 and groups == 252
                 and tasks == (("debt_sum", 250), ("fraction_sum", 120), ("linear_equation", 120), ("unit_conversion", 160)), "CLOSED_PROFILE_COUNTS")
        expected_counts = {split: {"total": count, "by_domain": {d: n for s, d, n in domains if s == split}} for split, count in counts}
        _require(metadata[0]["counts"] == expected_counts, "MANIFEST_COUNTS")
        return CorpusObservation(True, True, (), counts, domains, groups)
    except CorpusContractError as exc:
        return CorpusObservation(matched, False, (str(exc),))
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        return CorpusObservation(matched, False, ("MALFORMED_ARTIFACT",))
