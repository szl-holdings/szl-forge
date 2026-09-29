"""Bounded, source-only association repairs for owned/synthetic HTML fragments.

No submitted HTML, JavaScript, or model-generated code is ever executed here.
This deliberately small grammar is not a browser parser or a WCAG evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from html import unescape
from html.parser import HTMLParser
import re
from typing import Any
import unicodedata

MAX_SOURCE_BYTES = 32 * 1024
MAX_NODES = 256
MAX_CANDIDATES = 20

LIMITATIONS = [
    "Only explicit adjacent plain-text label/input associations in the supported HTML fragment grammar are checked.",
    "Rendered visibility, screen-reader behavior, keyboard usability, and intended label meaning require human evaluation.",
    "VERIFIED_SCOPED_REPAIR is not full accessibility, WCAG conformance, or a legal compliance certification.",
    "HTML is analyzed as text, never executed or fetched; no external resources are loaded.",
    "Source binding rejects stale proposals, but stateless calls do not provide a durable global replay ledger.",
]

_TAGS = frozenset({
    "form", "fieldset", "legend", "div", "p", "span", "label", "input",
    "button", "br", "small", "strong", "em", "main", "section", "article",
    "header", "footer", "h1", "h2", "h3",
})
_VOID = frozenset({"input", "br"})
_INLINE = frozenset({"span", "small", "strong", "em", "br", "label", "input", "button"})
_TEXT_CONTAINERS = frozenset({"span", "small", "strong", "em", "p", "legend", "h1", "h2", "h3"})
_GLOBAL_ATTRS = frozenset({"id", "title", "lang", "dir"})
_TAG_ATTRS = {
    "form": frozenset({"method", "action", "autocomplete", "novalidate"}),
    "label": frozenset({"for"}),
    "input": frozenset({
        "type", "name", "value", "placeholder", "autocomplete", "required",
        "disabled", "readonly", "checked", "multiple", "min", "max", "step",
        "minlength", "maxlength", "size", "pattern",
    }),
    "button": frozenset({"type", "name", "value", "disabled"}),
    "fieldset": frozenset({"disabled", "name"}),
}
_BOOLEAN = frozenset({"required", "disabled", "readonly", "checked", "multiple", "novalidate"})
_INPUT_TYPES = frozenset({
    "text", "email", "tel", "url", "password", "number", "search", "date",
    "time", "datetime-local", "month", "week", "checkbox", "radio", "range",
    "color", "file",
})
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_TAG_START = re.compile(r"<([A-Za-z][A-Za-z0-9]*)")
_ATTR = re.compile(r'''\s+([A-Za-z_:][A-Za-z0-9_.:-]*)(?:\s*=\s*("[^"]*"|'[^']*'))?''')
_END = re.compile(r"</([A-Za-z][A-Za-z0-9]*)\s*>\Z")


class _Unsupported(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str | None]
    start: int
    insertion: int
    parent: "_Node | None"
    children: list["_Node | str"] = field(default_factory=list)


class _FragmentParser(HTMLParser):
    """Reject tolerant-parser repairs and all constructs outside our grammar."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.line_starts = [0]
        self.line_starts.extend(match.end() for match in re.finditer("\n", source))
        self.root = _Node("#fragment", {}, 0, 0, None)
        self.stack = [self.root]
        self.nodes: list[_Node] = []
        self.node_count = 0

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def _count(self) -> None:
        self.node_count += 1
        if self.node_count > MAX_NODES:
            raise _Unsupported("NODE_LIMIT", f"At most {MAX_NODES} source nodes are supported.")

    def _start(self, tag: str, attrs: list[tuple[str, str | None]], self_closed: bool) -> None:
        self._count()
        if tag not in _TAGS:
            raise _Unsupported("UNSUPPORTED_ELEMENT", f"Element {tag!r} is outside the safe fragment grammar.")
        if self_closed and tag not in _VOID:
            raise _Unsupported("MALFORMED_HTML", "Only void elements may use self-closing syntax.")
        raw = self.get_starttag_text()
        match = _TAG_START.match(raw)
        if match is None:
            raise _Unsupported("MALFORMED_HTML", "Invalid start tag.")
        ending = "/>" if self_closed else ">"
        body = raw[match.end():-len(ending)]
        cursor = 0
        lexical: list[tuple[str, str | None]] = []
        while cursor < len(body):
            if body[cursor:].isspace():
                break
            item = _ATTR.match(body, cursor)
            if item is None:
                raise _Unsupported("MALFORMED_ATTRIBUTES", "Attribute values must be quoted and names separated by whitespace.")
            name, value = item.groups()
            name = name.lower()
            lexical.append((name, None if value is None else unescape(value[1:-1])))
            cursor = item.end()
        if lexical != attrs:
            raise _Unsupported("MALFORMED_ATTRIBUTES", "Parser and strict attribute tokenizer disagree.")
        names = [name for name, _ in attrs]
        if len(names) != len(set(names)):
            raise _Unsupported("DUPLICATE_ATTRIBUTE", "Duplicate attribute names are ambiguous.")
        allowed = _GLOBAL_ATTRS | _TAG_ATTRS.get(tag, frozenset())
        for name, value in attrs:
            if name not in allowed:
                raise _Unsupported("UNSUPPORTED_ATTRIBUTE", f"Attribute {name!r} is outside the safe fragment grammar.")
            if value is None and name not in _BOOLEAN:
                raise _Unsupported("MALFORMED_ATTRIBUTES", f"Attribute {name!r} requires a quoted value.")
            if value is not None and any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise _Unsupported("CONTROL_CHARACTER", "Control characters are not supported in attributes.")
            if name in {"id", "for"} and (value is None or _ID.fullmatch(value) is None):
                raise _Unsupported("UNSUPPORTED_IDENTIFIER", "Identifiers must be simple bounded ASCII identifiers.")
        attributes = dict(attrs)
        if attributes.get("action", "") != "":
            raise _Unsupported("EXTERNAL_ACTION", "Form actions are not accepted or fetched.")
        if tag == "input" and attributes.get("type", "text").lower() not in _INPUT_TYPES:
            raise _Unsupported("UNSUPPORTED_CONTROL", "Only explicitly labelable input types are supported.")
        if tag == "button" and attributes.get("type", "submit").lower() not in {"button", "submit", "reset"}:
            raise _Unsupported("UNSUPPORTED_CONTROL", "Unsupported button type.")
        parent = self.stack[-1]
        if parent.tag == "label":
            raise _Unsupported("AMBIGUOUS_LABEL", "Labels must contain plain text only; nested labels and controls need review.")
        if parent.tag == "button" and tag not in {"span", "small", "strong", "em", "br"}:
            raise _Unsupported("MALFORMED_HTML", "Interactive or block content inside a button is unsupported.")
        if any(node.tag == "button" for node in self.stack) and tag in {"button", "input", "label"}:
            raise _Unsupported("MALFORMED_HTML", "Nested interactive button content is unsupported.")
        if parent.tag in _TEXT_CONTAINERS and tag not in _INLINE:
            raise _Unsupported("MALFORMED_HTML", "Block content inside a text container is unsupported.")
        if tag == "form" and any(node.tag == "form" for node in self.stack):
            raise _Unsupported("MALFORMED_HTML", "Nested forms are unsupported.")
        if tag == "legend" and parent.tag != "fieldset":
            raise _Unsupported("MALFORMED_HTML", "A legend must belong directly to a fieldset.")
        start = self._offset()
        node = _Node(tag, attributes, start, start + match.end(), parent)
        parent.children.append(node)
        self.nodes.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs, False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs, True)

    def handle_endtag(self, tag: str) -> None:
        start = self._offset()
        end = self.source.find(">", start)
        raw = self.source[start:end + 1]
        if _END.fullmatch(raw) is None or len(self.stack) == 1 or self.stack[-1].tag != tag:
            raise _Unsupported("MALFORMED_HTML", "Closing tags must exactly match the explicitly opened element.")
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        if data:
            self._count()
            # A raw '<' left behind by HTMLParser can represent a truncated tag.
            original = self.source[self._offset():]
            if "<" in data and original.startswith("<"):
                raise _Unsupported("MALFORMED_HTML", "Unescaped or incomplete markup is unsupported.")
            self.stack[-1].children.append(data)

    def handle_comment(self, data: str) -> None:
        raise _Unsupported("UNSUPPORTED_COMMENT", "Comments are outside the explicit adjacency grammar.")

    def handle_decl(self, decl: str) -> None:
        raise _Unsupported("UNSUPPORTED_DOCUMENT", "Submit an HTML form fragment, not a document declaration.")

    def unknown_decl(self, data: str) -> None:
        raise _Unsupported("UNSUPPORTED_DOCUMENT", "Unknown declarations are unsupported.")

    def handle_pi(self, data: str) -> None:
        raise _Unsupported("UNSUPPORTED_DOCUMENT", "Processing instructions are unsupported.")


def _digest(source: Any) -> str | None:
    if type(source) is not str:
        return None
    try:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        return None


def _has_invisible_controls(value: str) -> bool:
    return any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} and char not in "\t\r\n" for char in value)


def _inspect(source: str) -> tuple[dict[str, Any], dict[str, _Node]]:
    digest = _digest(source)
    report: dict[str, Any] = {
        "source_sha256": digest, "status": "UNSUPPORTED", "candidates": [],
        "findings": [], "limitations": list(LIMITATIONS),
    }
    try:
        if digest is None:
            raise _Unsupported("INVALID_SOURCE", "Source must be a valid UTF-8 encodable string.")
        if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise _Unsupported("SOURCE_LIMIT", f"Source exceeds {MAX_SOURCE_BYTES} UTF-8 bytes.")
        if _has_invisible_controls(source):
            raise _Unsupported("CONTROL_CHARACTER", "Source contains unsupported control characters.")
        parser = _FragmentParser(source)
        parser.feed(source)
        parser.close()
        if len(parser.stack) != 1:
            raise _Unsupported("MALFORMED_HTML", "Every non-void element requires a matching closing tag.")
        identifiers: dict[str, _Node] = {}
        for node in parser.nodes:
            identifier = node.attrs.get("id")
            if identifier is not None:
                if identifier in identifiers:
                    raise _Unsupported("DUPLICATE_ID", "Every id must be unique in the entire submitted fragment.")
                identifiers[identifier] = node
        labels = [node for node in parser.nodes if node.tag == "label"]
        inputs = [node for node in parser.nodes if node.tag == "input"]
        association: dict[str, _Node] = {}
        candidates: list[dict[str, Any]] = []
        selected_nodes: dict[str, _Node] = {}
        for label in labels:
            label_text = " ".join("".join(part for part in label.children if isinstance(part, str)).split())
            if not label_text or _has_invisible_controls(label_text):
                raise _Unsupported("AMBIGUOUS_LABEL", "An empty label or label with invisible control characters requires review.")
            control_id = label.attrs.get("for")
            if control_id is not None:
                target = identifiers.get(control_id)
                if target is None or target.tag != "input":
                    raise _Unsupported("AMBIGUOUS_LABEL", "Existing label targets must identify an input in the fragment.")
            else:
                siblings = label.parent.children
                index = next(index for index, item in enumerate(siblings) if item is label) + 1
                while index < len(siblings) and isinstance(siblings[index], str) and siblings[index].isspace():
                    index += 1
                if index == len(siblings) or not isinstance(siblings[index], _Node) or siblings[index].tag != "input":
                    raise _Unsupported("AMBIGUOUS_LABEL", "An unbound label must immediately precede its input under the same parent.")
                target = siblings[index]
                control_id = target.attrs.get("id")
                if control_id is None:
                    raise _Unsupported("MISSING_CONTROL_ID", "The adjacent input needs an existing unique id; ids are never invented.")
                candidate_id = "label-" + hashlib.sha256(f"{digest}:{label.start}:{control_id}".encode()).hexdigest()[:24]
                candidates.append({
                    "id": candidate_id, "label_text": label_text, "control_id": control_id,
                    "label_start": label.start, "control_start": target.start,
                    "repair": "add_explicit_for_attribute",
                })
                selected_nodes[candidate_id] = label
            if control_id in association:
                raise _Unsupported("AMBIGUOUS_LABEL", "Multiple labels for one control require human review.")
            association[control_id] = label
        for control in inputs:
            if control.attrs.get("id") not in association:
                raise _Unsupported("MISSING_EXPLICIT_LABEL", "An input without an explicit or adjacent plain-text label requires review.")
        if len(candidates) > MAX_CANDIDATES:
            raise _Unsupported("CANDIDATE_LIMIT", f"At most {MAX_CANDIDATES} associations may be considered together.")
        report["status"] = "SUPPORTED"
        report["candidates"] = candidates
        report["findings"] = [{
            "code": "MISSING_EXPLICIT_ASSOCIATION", "detail": "The adjacent text label lacks an explicit for attribute.",
            "candidate_id": candidate["id"],
        } for candidate in candidates]
        return report, selected_nodes
    except _Unsupported as error:
        report["findings"] = [{"code": error.code, "detail": error.detail}]
        return report, {}
    except (AssertionError, ValueError, RecursionError):
        report["findings"] = [{"code": "MALFORMED_HTML", "detail": "HTML could not be parsed within the supported fragment grammar."}]
        return report, {}


def analyze(source: str) -> dict[str, Any]:
    """Return source-bound candidate associations without executing user content."""
    return _inspect(source)[0]


def deterministic_proposal(source: str) -> dict[str, Any]:
    """Select all eligible associations; this is a rules baseline, not a model."""
    report = analyze(source)
    return {
        "source_sha256": report["source_sha256"],
        "bindings": [{"candidate_id": candidate["id"]} for candidate in report["candidates"]],
        "methodology": "deterministic",
    }


def apply_proposal(source: str, proposal: dict[str, Any]) -> dict[str, Any]:
    """Apply only source-bound candidate selections through a closed patch schema.

    Repeated calls with the exact original source remain deterministic. Reusing a
    proposal against the repaired source is rejected by its source digest.
    """
    analysis, nodes = _inspect(source)
    digest = analysis["source_sha256"]
    checks = {
        "supported_source": analysis["status"] == "SUPPORTED", "proposal_schema": False,
        "source_binding": False, "candidate_binding": False,
        "source_preserved": True, "post_analysis": False,
        "human_evaluation_required": True,
    }
    result: dict[str, Any] = {
        "state": "REVIEW_REQUIRED", "source_sha256": digest, "output_sha256": digest,
        "output_html": source if type(source) is str else "", "applied_bindings": [],
        "checks": checks, "findings": list(analysis["findings"]),
        "limitations": list(LIMITATIONS),
    }

    def refuse(code: str, detail: str) -> dict[str, Any]:
        result["findings"].append({"code": code, "detail": detail})
        return result

    if type(proposal) is not dict or set(proposal) != {"source_sha256", "bindings", "methodology"}:
        return refuse("PROPOSAL_SCHEMA", "A proposal must have exactly source_sha256, bindings, and methodology.")
    if (type(proposal["source_sha256"]) is not str
            or type(proposal["methodology"]) is not str
            or proposal["methodology"] not in {"deterministic", "model"}
            or type(proposal["bindings"]) is not list
            or len(proposal["bindings"]) > MAX_CANDIDATES):
        return refuse("PROPOSAL_SCHEMA", "Proposal field types, methodology, or binding count are invalid.")
    ids: list[str] = []
    for binding in proposal["bindings"]:
        if type(binding) is not dict or set(binding) != {"candidate_id"} or type(binding["candidate_id"]) is not str:
            return refuse("PROPOSAL_SCHEMA", "Each binding must contain only a string candidate_id; executable code is not accepted.")
        ids.append(binding["candidate_id"])
    checks["proposal_schema"] = True
    if digest is None or proposal["source_sha256"] != digest:
        return refuse("STALE_SOURCE", "The proposal does not match these exact source bytes.")
    checks["source_binding"] = True
    if not checks["supported_source"]:
        return result
    if len(ids) != len(set(ids)) or any(candidate_id not in nodes for candidate_id in ids):
        return refuse("CANDIDATE_BINDING", "Candidate selections must be unique and come from this source analysis.")
    checks["candidate_binding"] = True
    if not ids:
        checks["post_analysis"] = True
        result["state"] = "NO_CHANGE"
        return result
    candidates = {candidate["id"]: candidate for candidate in analysis["candidates"]}
    insertions = sorted((nodes[candidate_id].insertion, f' for="{candidates[candidate_id]["control_id"]}"') for candidate_id in ids)
    chunks: list[str] = []
    cursor = 0
    for offset, insertion in insertions:
        chunks.extend((source[cursor:offset], insertion))
        cursor = offset
    chunks.append(source[cursor:])
    output = "".join(chunks)
    after = analyze(output)
    checks["source_preserved"] = "".join(chunks[::2]) == source
    checks["post_analysis"] = after["status"] == "SUPPORTED" and len(after["candidates"]) == len(analysis["candidates"]) - len(ids)
    if not checks["source_preserved"] or not checks["post_analysis"]:
        return refuse("POSTCONDITION_FAILED", "The proposed association failed a scoped preservation or post-analysis check.")
    result.update({
        "state": "VERIFIED_SCOPED_REPAIR", "output_sha256": _digest(output),
        "output_html": output,
        "applied_bindings": [candidates[candidate_id] for candidate_id in ids],
        "findings": [{"code": "SCOPED_ASSOCIATION_VERIFIED", "detail": "Selected explicit label associations were added; human usability evaluation remains required."}],
    })
    return result
