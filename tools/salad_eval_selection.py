"""Select a reproducible, exact-match-filtered evaluation residual.

No network, model loading, training, promotion, or source mutation is performed.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import unicodedata

NORMALIZER_ID = "szl.prompt.nfkc-casefold-whitespace/v1"
SELECTOR_ID = "szl.salad-eval-selection/v1"


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def value_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("nonstandard JSON number")


def _message_identity(row):
    if not isinstance(row, dict) or set(row) != {"messages", "source"}:
        raise ValueError("expected exactly messages/source row fields")
    if not isinstance(row["source"], str) or not row["source"].strip():
        raise ValueError("source must be a nonempty string")
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    transcript = []
    users = []
    prefixes = set()
    expected = "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not {"role", "content"} <= set(message) or set(message) - {"role", "content", "tool_calls"}:
            raise ValueError("unsupported or ignored message metadata")
        if "tool_calls" in message and message["tool_calls"] != []:
            raise ValueError("only exactly empty tool_calls metadata is supported")
        role, content = message["role"], message["content"]
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        if index == 0 and role == "system":
            transcript.append({"role": role, "content": normalize(content)})
            continue
        if role != expected:
            raise ValueError("expected alternating user/assistant roles")
        normalized = normalize(content)
        if not normalized:
            raise ValueError("empty user message or supervised assistant target")
        if role == "user":
            users.append(normalized)
            expected = "assistant"
        else:
            if not users:
                raise ValueError("assistant target has no prior user input")
            prefixes.add(value_hash(transcript))
            expected = "user"
        transcript.append({"role": role, "content": normalized})
    if not users or expected != "user" or transcript[-1]["role"] != "assistant":
        raise ValueError("conversation must end with a supervised assistant target")
    return {
        "user_turns": {hashlib.sha256(x.encode("utf-8")).hexdigest() for x in users},
        # Selection keys contain user input only, never assistant answer content.
        "user_context": value_hash(users),
        "full_prompt": value_hash(transcript[:-1]),
        "target_prefixes": prefixes,
    }


def iter_rows(path, expected_sha256=None):
    record_index = 0
    consumed = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for physical_line, raw in enumerate(stream, 1):
            consumed.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(
                    raw.decode("utf-8"), object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
                identity = _message_identity(row)
            except (UnicodeError, ValueError, TypeError) as exc:
                # Do not echo JSON data, prompt content, or parser excerpts.
                reason = "invalid JSON or message schema"
                if type(exc) is ValueError and not isinstance(exc, json.JSONDecodeError):
                    reason = str(exc)
                raise ValueError(f"physical line {physical_line}: {reason}") from None
            yield {
                "physical_line_1based": physical_line,
                "record_index_0based": record_index,
                "raw_line_sha256": hashlib.sha256(raw).hexdigest(),
                "source": row["source"],
                "raw": raw,
                **identity,
            }
            record_index += 1
    if expected_sha256 is not None and consumed.hexdigest() != expected_sha256:
        raise ValueError("consumed input bytes do not match immutable file hash")


def _verify_inputs(spec, train_path, eval_path):
    if spec["normalization"] != {
        "id": NORMALIZER_ID, "unicode_data_version": unicodedata.unidata_version,
    }:
        raise ValueError("normalizer or Unicode database version mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", spec["dataset"]["revision"]):
        raise ValueError("dataset revision must be an immutable commit")
    if Path(train_path).resolve() == Path(eval_path).resolve():
        raise ValueError("training and evaluation input paths must differ")
    for split, path in (("train", train_path), ("eval", eval_path)):
        expected = spec["dataset"]["files"][split]
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"{split} file hash does not match immutable input")
        if type(expected["rows"]) is not int or expected["rows"] <= 0:
            raise ValueError(f"{split} expected row count must be a positive integer")


def build_selection(spec, train_path, eval_path):
    _verify_inputs(spec, train_path, eval_path)
    training_turns, training_contexts, training_prompts, training_prefixes = (
        set(), set(), set(), set()
    )
    train_sources = collections.Counter()
    train_rows = 0
    for row in iter_rows(train_path, spec["dataset"]["files"]["train"]["sha256"]):
        train_rows += 1
        train_sources[row["source"]] += 1
        training_turns.update(row["user_turns"])
        training_contexts.add(row["user_context"])
        training_prompts.add(row["full_prompt"])
        training_prefixes.update(row["target_prefixes"])
    if train_rows != spec["dataset"]["files"]["train"]["rows"]:
        raise ValueError("train row count differs from immutable manifest")

    seen_eval_contexts = set()
    original_sources = collections.Counter()
    overlap_sources = collections.Counter()
    duplicate_sources = collections.Counter()
    retained_sources = collections.Counter()
    selected = []
    subset_hash = hashlib.sha256()
    subset_bytes = 0
    eval_rows = overlap_rows = duplicate_rows = 0
    retained_turns, retained_prompts, retained_prefixes = set(), set(), set()
    for row in iter_rows(eval_path, spec["dataset"]["files"]["eval"]["sha256"]):
        eval_rows += 1
        source = row["source"]
        original_sources[source] += 1
        # A row is counted once even when several of its turns overlap.
        if row["user_turns"] & training_turns:
            overlap_rows += 1
            overlap_sources[source] += 1
            continue
        if row["user_context"] in seen_eval_contexts:
            duplicate_rows += 1
            duplicate_sources[source] += 1
            continue
        seen_eval_contexts.add(row["user_context"])
        retained_sources[source] += 1
        retained_turns.update(row["user_turns"])
        retained_prompts.add(row["full_prompt"])
        retained_prefixes.update(row["target_prefixes"])
        subset_hash.update(row["raw"])
        subset_bytes += len(row["raw"])
        selected.append({
            "physical_line_1based": row["physical_line_1based"],
            "record_index_0based": row["record_index_0based"],
            "raw_line_sha256": row["raw_line_sha256"],
            "user_context_sha256": row["user_context"],
            "source": source,
        })
    if eval_rows != spec["dataset"]["files"]["eval"]["rows"]:
        raise ValueError("eval row count differs from immutable manifest")
    intersections = {
        "user_turns": len(retained_turns & training_turns),
        "user_contexts": len(seen_eval_contexts & training_contexts),
        "full_prompts": len(retained_prompts & training_prompts),
        "assistant_target_prefixes": len(retained_prefixes & training_prefixes),
    }
    if any(intersections.values()):
        raise ValueError("retained selection violates exact separation invariant")
    if not selected:
        raise ValueError("no evaluation rows remain after filtering")
    selection = {
        "schema": SELECTOR_ID,
        "dataset": spec["dataset"],
        "normalization": spec["normalization"],
        "selection_policy": {
            "train": "all original rows, unchanged",
            "eval_exclusion": "any complete normalized user turn in original training",
            "eval_deduplication": "ordered complete normalized user turns; first original row",
            "answer_content_used_for_selection": False,
            "raw_line_hash_includes_newline": True,
            "message_schema": "role/content and optional exactly-empty tool_calls only",
        },
        "selected_eval_rows": selected,
    }
    selection_hash = value_hash(selection)
    manifest = {
        "schema": "szl.salad-candidate-data-manifest/v1",
        "status": "EXACT_MATCH_FILTERED_CANDIDATE_REVIEW_REQUIRED",
        "training_qualified": False,
        "independent_generalization": "NOT_ESTABLISHED",
        "launch_authorized": False,
        "dataset": spec["dataset"],
        "normalization": spec["normalization"],
        "selector": {
            "id": SELECTOR_ID, "path": "tools/salad_eval_selection.py",
            "sha256": file_hash(__file__),
        },
        "selection_canonical_sha256": selection_hash,
        "ordered_raw_line_hashes_sha256": value_hash(
            [x["raw_line_sha256"] for x in selected]
        ),
        "materialized_eval": {
            "rows": len(selected), "bytes": subset_bytes,
            "sha256": subset_hash.hexdigest(),
        },
        "counts": {
            "train_rows_preserved": train_rows,
            "original_eval_rows": eval_rows,
            "excluded_eval_rows_for_training_turn_overlap": overlap_rows,
            "excluded_residual_eval_duplicate_rows": duplicate_rows,
            "selected_eval_rows": len(selected),
        },
        "source_counts": {
            "train_original": dict(sorted(train_sources.items())),
            "eval_original": dict(sorted(original_sources.items())),
            "eval_overlap_removed": dict(sorted(overlap_sources.items())),
            "eval_duplicate_removed": dict(sorted(duplicate_sources.items())),
            "eval_selected": dict(sorted(retained_sources.items())),
        },
        "retained_intersections": intersections,
        "candidate_plan": spec.get("candidate_plan", {}),
        "limitations": [
            "Normalized equality is conservative; code/math distinctions may collapse.",
            "Semantic near-duplicates and source-family separation are not established.",
            "Licenses, consent, PII handling, and pretraining contamination are not verified here.",
            "This residual is not a new unseen benchmark for previously exposed adapters.",
            "Filtering now does not erase prior adapter exposure or qualify production behavior.",
            "No dataset, candidate model, GPU job, or release was changed by this selector.",
        ],
    }
    return selection, manifest


def _validate_materialization(selection, manifest):
    if selection.get("schema") != SELECTOR_ID:
        raise ValueError("unsupported selection schema")
    if selection.get("dataset") != manifest.get("dataset"):
        raise ValueError("selection and manifest dataset differ")
    normalization = {"id": NORMALIZER_ID, "unicode_data_version": unicodedata.unidata_version}
    if selection.get("normalization") != normalization or manifest.get("normalization") != normalization:
        raise ValueError("selection and manifest normalizer differ")
    if value_hash(selection) != manifest.get("selection_canonical_sha256"):
        raise ValueError("selection is not bound to candidate manifest")
    if manifest.get("selector", {}).get("sha256") != file_hash(__file__):
        raise ValueError("selector source differs from candidate manifest")
    rows = selection.get("selected_eval_rows")
    expected = manifest.get("materialized_eval", {})
    if not isinstance(rows, list) or not rows or type(expected.get("rows")) is not int or len(rows) != expected["rows"]:
        raise ValueError("selected row count is invalid")
    previous_line = 0
    previous_record = -1
    contexts = set()
    sources = collections.Counter()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "physical_line_1based", "record_index_0based", "raw_line_sha256",
            "user_context_sha256", "source",
        }:
            raise ValueError("selected row identity fields are invalid")
        line, record = row["physical_line_1based"], row["record_index_0based"]
        if type(line) is not int or type(record) is not int or line <= previous_line or record <= previous_record:
            raise ValueError("selected row identities must be unique and in original order")
        if line < 1 or record < 0 or record >= line:
            raise ValueError("selected row index bounds are invalid")
        for key in ("raw_line_sha256", "user_context_sha256"):
            if not isinstance(row[key], str) or not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("selected row digest is invalid")
        if row["user_context_sha256"] in contexts:
            raise ValueError("selected evaluation prompt identities are not unique")
        contexts.add(row["user_context_sha256"])
        if not isinstance(row["source"], str) or not row["source"]:
            raise ValueError("selected source label is invalid")
        sources[row["source"]] += 1
        previous_line, previous_record = line, record
    if manifest.get("counts", {}).get("selected_eval_rows") != len(rows):
        raise ValueError("candidate manifest selected count differs")
    if dict(sorted(sources.items())) != manifest.get("source_counts", {}).get("eval_selected"):
        raise ValueError("candidate manifest selected source counts differ")
    if value_hash([x["raw_line_sha256"] for x in rows]) != manifest.get("ordered_raw_line_hashes_sha256"):
        raise ValueError("ordered selected row digest differs")
    if type(expected.get("bytes")) is not int or expected["bytes"] <= 0:
        raise ValueError("materialized byte count is invalid")
    return rows


def materialize_eval(eval_path, destination, selection, manifest):
    rows = _validate_materialization(selection, manifest)
    destination = Path(destination)
    if destination.exists() or destination.resolve() == Path(eval_path).resolve():
        raise ValueError("materialization destination must be new")
    expected_source_hash = selection["dataset"]["files"]["eval"]["sha256"]
    if file_hash(eval_path) != expected_source_hash:
        raise ValueError("eval source changed before materialization")
    wanted = {x["physical_line_1based"]: x for x in rows}
    digest = hashlib.sha256()
    selected_count = byte_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        target = destination.open("xb")
        created = True
        with target:
            for row in iter_rows(eval_path, expected_source_hash):
                identity = wanted.get(row["physical_line_1based"])
                if identity is None:
                    continue
                if any(row[key] != identity[key] for key in (
                    "record_index_0based", "raw_line_sha256", "source"
                )) or row["user_context"] != identity["user_context_sha256"]:
                    raise ValueError("immutable selected row identity changed")
                target.write(row["raw"])
                digest.update(row["raw"])
                byte_count += len(row["raw"])
                selected_count += 1
        expected = manifest["materialized_eval"]
        if selected_count != expected["rows"] or byte_count != expected["bytes"]:
            raise ValueError("materialized row or byte count mismatch")
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("materialized evaluation hash mismatch")
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


def write_receipts(output_dir, selection, manifest, protected_paths=()):
    output_dir = Path(output_dir)
    protected = {Path(path).resolve() for path in protected_paths}
    for name in ("selection.json", "candidate-manifest.json"):
        if (output_dir / name).resolve() in protected:
            raise ValueError("receipt output must not overwrite an input")
    # Each receipt set gets a fresh directory; historical receipts cannot be replaced.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("selection.json", selection), ("candidate-manifest.json", manifest)):
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materialize-eval", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    selection, manifest = build_selection(spec, args.train, args.eval)
    # Both inputs are fully checked before receipt output is written.
    write_receipts(args.output_dir, selection, manifest, (args.train, args.eval, args.spec))
    if args.materialize_eval:
        if args.materialize_eval.resolve() == args.train.resolve():
            raise ValueError("must not materialize over training input")
        materialize_eval(args.eval, args.materialize_eval, selection, manifest)
    print(json.dumps({
        "status": manifest["status"], "counts": manifest["counts"],
        "retained_intersections": manifest["retained_intersections"],
        "selection_canonical_sha256": manifest["selection_canonical_sha256"],
        "materialized_eval": manifest["materialized_eval"],
        "ordered_raw_line_hashes_sha256": manifest["ordered_raw_line_hashes_sha256"],
        "launch_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
