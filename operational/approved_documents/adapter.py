"""Local, owner-declared document intake and exact-quote verification; no inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

PACKET_SCHEMA = "szl.approved-document-packet/v1"
MANIFEST_SCHEMA = "szl.approved-document-manifest/v1"
RECEIPT_SCHEMA = "szl.document-span-verification/v1"
PREVIEW_SOURCE = "a220bb8f33a9fb2825e3978821e9a96659ed470e"
MAX_JSON_BYTES = 128 * 1024
MAX_JSON_DEPTH = 16
MAX_DOCUMENT_BYTES = 64000
MAX_QUOTED_BYTES = 64000
MAX_SPANS = 16
AUTHORITY = {
    "clinicalUseAuthorized": False,
    "trainingAuthorized": False,
    "publicationAuthorized": False,
    "executionAuthorized": False,
}
DECLARATIONS = {
    "rightsBasis", "rightsConfirmedByOwner", "sensitivity", "permittedUse"
}
RIGHTS = {"OWNER_AUTHORED", "LICENSE_PERMITS_LOCAL_PROCESSING", "EXPLICIT_PERMISSION"}
SHA = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
SOURCE_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.(txt|md)\Z")
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10))}


class DocumentError(ValueError):
    """Invalid input is never evidence admission."""


def canonical(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise DocumentError("Value is not bounded UTF-8 JSON") from exc


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact(value: object, keys: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise DocumentError(f"Invalid {label} fields")
    return value


def text(value: object, maximum: int, label: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise DocumentError(f"Invalid {label} text")
    # Permit ordinary line breaks and tabs, but not controls or lone surrogates.
    if any((ord(c) < 32 and c not in "\n\r\t") or 127 <= ord(c) <= 159
           or 0xD800 <= ord(c) <= 0xDFFF
           for c in value):
        raise DocumentError(f"Invalid {label} encoding")
    return value


def local_path(path: Path) -> Path:
    # Explicit local files only: no network shares, ADS, traversal, symlink or junction.
    raw = os.fspath(path)
    if raw.startswith(("\\\\", "//")) or ".." in Path(raw).parts:
        raise DocumentError("Only direct local paths are accepted")
    drive, tail = os.path.splitdrive(raw)
    if drive and not tail.startswith(("/", "\\")):
        raise DocumentError("Drive-relative paths are not accepted")
    # Windows abspath normalizes trailing dots/spaces, so inspect the raw parts first.
    for part in Path(raw).parts:
        if part.endswith((".", " ")) or part.split(".")[0].upper() in RESERVED_NAMES:
            raise DocumentError("Reserved or ambiguous local filename")
    absolute = Path(os.path.abspath(path))
    if ":" in str(absolute)[2:]:
        raise DocumentError("Alternate stream paths are not accepted")
    for part in absolute.parts[1:]:
        if part.endswith((".", " ")) or part.split(".")[0].upper() in RESERVED_NAMES:
            raise DocumentError("Reserved or ambiguous local filename")
    return absolute


def check_path(path: Path, *, directory: bool = False) -> tuple[Path, os.stat_result]:
    absolute = local_path(path)
    observed = None
    for component in (absolute, *absolute.parents):
        metadata = os.lstat(component)
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & REPARSE:
            raise DocumentError("Symlinks and reparse points are not accepted")
        if component == absolute:
            observed = metadata
    if observed is None or not (stat.S_ISDIR(observed.st_mode) if directory
                                else stat.S_ISREG(observed.st_mode)):
        raise DocumentError("Expected a regular local file or directory")
    return absolute, observed


def read_bytes(path: Path, limit: int) -> bytes:
    absolute, before = check_path(path)
    if not 0 < before.st_size <= limit:
        raise DocumentError("Input exceeds the permitted size or is empty")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(absolute, flags), "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
            raise DocumentError("Input changed while opening")
        if not stat.S_ISREG(opened.st_mode) or getattr(opened, "st_file_attributes", 0) & REPARSE:
            raise DocumentError("Input is not a regular file")
        raw = handle.read(limit + 1)
        after = os.fstat(handle.fileno())
    if (len(raw) > limit or len(raw) != opened.st_size or
            (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise DocumentError("Input changed while reading")
    return raw


def parse_json(raw: bytes) -> dict:
    if not 0 < len(raw) <= MAX_JSON_BYTES:
        raise DocumentError("Invalid JSON size")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DocumentError("Duplicate JSON field")
            result[key] = value
        return result

    def reject_number(_):
        raise DocumentError("Floating point and non-finite JSON values are not accepted")

    def integer(value):
        if len(value) > 10:
            raise DocumentError("Oversized JSON integer")
        return int(value)

    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                            parse_constant=reject_number, parse_float=reject_number,
                            parse_int=integer)
        if type(result) is not dict:
            raise DocumentError("JSON must contain an object")
        pending = [(result, 1)]
        while pending:
            value, depth = pending.pop()
            if depth > MAX_JSON_DEPTH:
                raise DocumentError("JSON nesting exceeds the permitted depth")
            if type(value) is dict:
                pending.extend((child, depth + 1) for child in value.values())
            elif type(value) is list:
                pending.extend((child, depth + 1) for child in value)
        canonical(result)
        return result
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise DocumentError("Invalid strict JSON") from exc


def load_json(path: Path) -> dict:
    return parse_json(read_bytes(path, MAX_JSON_BYTES))


def declarations(value: object) -> dict:
    value = exact(value, DECLARATIONS, "owner declarations")
    if (any(type(value[key]) is not str for key in DECLARATIONS - {"rightsConfirmedByOwner"})
            or value["rightsBasis"] not in RIGHTS or value["rightsConfirmedByOwner"] is not True
            or value["sensitivity"] != "PUBLIC_NON_SENSITIVE"
            or value["permittedUse"] != "LOCAL_EVIDENCE_REVIEW"):
        raise DocumentError("Explicit permitted nonsensitive owner declarations required")
    return dict(value)


def make_packet(document_id: str, title: str, declared: dict,
                question: str, context: str) -> dict:
    if type(document_id) is not str or IDENTIFIER.fullmatch(document_id) is None:
        raise DocumentError("Invalid document ID")
    title = text(title, 160, "title")
    question = text(question, 512, "question")
    context = text(context, 16000, "context")
    body = {"question": question, "context": context}
    raw = context.encode("utf-8")
    if len(raw) > MAX_DOCUMENT_BYTES or len(canonical(body)) > 65536:
        raise DocumentError("Document or request exceeds Evidence Lab input limits")
    packet = {
        "schema": PACKET_SCHEMA,
        "state": "OWNER_DECLARED_LOCAL_REQUEST_PREPARED",
        "document": {"id": document_id, "title": title, "sha256": digest(raw),
                     "bytes": len(raw), "codepoints": len(context)},
        "ownerDeclarations": declarations(declared),
        "provenance": "DECLARED_NOT_INDEPENDENTLY_VERIFIED",
        "sourceContentTrusted": False,
        "rightsIndependentlyVerified": False,
        "runtimeWitnessed": False,
        "request": {"method": "POST", "path": "/api/answer-context",
                    "headers": {"Content-Type": "application/json", "X-SZL-Preview": "1"},
                    "body": body, "bodySha256": digest(canonical(body)),
                    "contractSourceRevision": PREVIEW_SOURCE},
        "authority": dict(AUTHORITY),
    }
    packet["packetSha256"] = digest(canonical(packet))
    return packet


def prepare(manifest_path: Path) -> dict:
    manifest = exact(load_json(manifest_path), {
        "schema", "documentId", "title", "sourceFile", "sourceSha256", "question",
        "ownerDeclarations"
    }, "manifest")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise DocumentError("Invalid manifest schema")
    source = manifest["sourceFile"]
    if type(source) is not str or SOURCE_FILE.fullmatch(source) is None:
        raise DocumentError("sourceFile must be a simple .txt or .md basename")
    # Windows device names remain reserved even when followed by an extension.
    if source.split(".")[0].upper() in RESERVED_NAMES:
        raise DocumentError("Reserved source filename")
    claimed = manifest["sourceSha256"]
    if type(claimed) is not str or SHA.fullmatch(claimed) is None:
        raise DocumentError("Expected full lowercase source SHA-256")
    raw = read_bytes(manifest_path.absolute().parent / source, MAX_DOCUMENT_BYTES)
    if digest(raw) != claimed:
        raise DocumentError("Source digest differs from owner manifest")
    try:
        context = raw.decode("utf-8")
    except UnicodeError as exc:
        raise DocumentError("Source must be UTF-8 text") from exc
    return make_packet(manifest["documentId"], manifest["title"],
                       manifest["ownerDeclarations"], manifest["question"], context)


def validate_packet(packet: object) -> dict:
    value = exact(packet, {"schema", "state", "document", "ownerDeclarations", "provenance",
                          "sourceContentTrusted", "rightsIndependentlyVerified", "runtimeWitnessed",
                          "request", "authority", "packetSha256"}, "packet")
    doc = exact(value["document"], {"id", "title", "sha256", "bytes", "codepoints"}, "document")
    request = exact(value["request"], {"method", "path", "headers", "body", "bodySha256",
                                       "contractSourceRevision"}, "request")
    body = exact(request["body"], {"question", "context"}, "request body")
    expected = make_packet(doc["id"], doc["title"], value["ownerDeclarations"],
                           body["question"], body["context"])
    # Canonical bytes also distinguish bool from integer, unlike Python equality.
    if canonical(value) != canonical(expected):
        raise DocumentError("Packet content, authority or digest binding differs")
    return expected


def verify_spans(packet: object, candidate: object) -> dict:
    packet = validate_packet(packet)
    context = packet["request"]["body"]["context"]
    verified = []
    reason = "NO_CITATIONS"
    try:
        candidate = exact(candidate, {"sourceSha256", "spans"}, "candidate citations")
        if candidate["sourceSha256"] != packet["document"]["sha256"]:
            raise DocumentError("Source digest differs")
        spans = candidate["spans"]
        if type(spans) is not list or len(spans) > MAX_SPANS:
            raise DocumentError("Invalid span count")
        seen = set()
        quoted_bytes = 0
        for span in spans:
            span = exact(span, {"start", "end", "quote"}, "citation span")
            start, end = span["start"], span["end"]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(context) or (start, end) in seen):
                raise DocumentError("Invalid or duplicate codepoint span")
            quote = text(span["quote"], 16000, "quote")
            if quote != context[start:end]:
                raise DocumentError("Quote differs from the exact source span")
            quoted_bytes += len(quote.encode("utf-8"))
            if quoted_bytes > MAX_QUOTED_BYTES:
                raise DocumentError("Cumulative citation text exceeds the permitted size")
            seen.add((start, end))
            verified.append(dict(span))
        if verified:
            reason = "EXACT_SOURCE_SPANS_MATCH"
    except DocumentError:
        verified = []
        reason = "INVALID_CITATION_BINDING"
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "state": "VERIFIED_SPANS_ONLY" if verified else "ABSTAIN",
        "reason": reason,
        "packetSha256": packet["packetSha256"],
        "sourceSha256": packet["document"]["sha256"],
        "offsetUnit": "UNICODE_CODEPOINTS_END_EXCLUSIVE",
        "spans": verified,
        "semanticSupportVerified": False,
        "answerTruthVerified": False,
        "independentlyAttested": False,
        "authority": dict(AUTHORITY),
    }
    receipt["receiptSha256"] = digest(canonical(receipt))
    return receipt


def write_new(path: Path, value: dict) -> None:
    # Parent must already be a trusted, stable local directory. Never replace a file.
    absolute = local_path(path)
    check_path(absolute.parent, directory=True)
    raw = canonical(value) + b"\n"
    if len(raw) > MAX_JSON_BYTES:
        raise DocumentError("Output exceeds the reloadable JSON size limit")
    parse_json(raw)
    with absolute.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    intake = commands.add_parser("prepare", help="Prepare, but never send, a local request")
    intake.add_argument("--manifest", type=Path, required=True)
    check = commands.add_parser("verify", help="Verify proposed exact source spans only")
    check.add_argument("--packet", type=Path, required=True)
    check.add_argument("--citations", type=Path, required=True)
    for command in (intake, check):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (prepare(args.manifest) if args.command == "prepare" else
                  verify_spans(load_json(args.packet), load_json(args.citations)))
        write_new(args.output, result)
    except (DocumentError, OSError) as exc:
        print(f"Document evidence refused: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(result["state"])
    return 3 if result["state"] == "ABSTAIN" else 0


if __name__ == "__main__":
    raise SystemExit(main())
