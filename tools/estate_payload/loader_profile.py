# SPDX-License-Identifier: Apache-2.0
"""Pure comparison of a bounded explicit loader descriptor; never a loader.

``observe_loader_profile`` accepts immutable UTF-8 JSON bytes and independently
supplied expected raw SHA256, candidate/recipe snapshot digests and purpose. It
does no file, network, framework, environment or runtime access. Raw bytes are
not normalized: the external digest covers the exact supplied representation.

All objects have closed fields and exact types. Numbers, duplicate keys and
nonfinite constants are rejected. Base and tokenizer bindings are independently
explicit (they may deliberately name the same repo/revision); no family or plan
is inferred from a repository name. Factory identifiers below are a closed
vocabulary, not dynamic imports or a declaration of installed runtime support.
Only no-offload placement is supported by this first descriptor-only slice.

The result is immutable comparison evidence under a trusted Python process,
not a capability or hostile-process boundary. Caller-provided expectations do
not establish signing trust, authenticate a runtime, verify artifact bytes or
observe actual class/precision/placement. All authorization flags stay false;
no existing candidate, profile, runtime lock or signed artifact is assigned or
changed. Future loading/returned-model validation remains a separate gap.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re


SCHEMA = "szl.loader-profile/v1"
MAX_PROFILE_BYTES = 16 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_CLASS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_MODEL_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}")
_PURPOSES = frozenset({"train", "evaluate", "export"})
_FLOAT_DTYPES = frozenset({"float32", "float16", "bfloat16"})
_LOADER_PAIRS = frozenset({
    ("transformers", "AutoModelForCausalLM"),
    ("transformers", "AutoModelForImageTextToText"),
    ("transformers", "AutoModelForVision2Seq"),
    ("unsloth", "FastLanguageModel"),
})


class LoaderProfileError(ValueError):
    """Stable, value-free rejection code; no provider or descriptor disclosure."""


def _require(condition, code):
    if condition is not True:
        raise LoaderProfileError(code)


def _fields(value, names, code):
    _require(type(value) is dict and set(value) == set(names), code)


def _matches(value, pattern, code):
    _require(type(value) is str and pattern.fullmatch(value) is not None, code)
    return value


def _choice(value, choices, code):
    _require(type(value) is str and value in choices, code)
    return value


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "PROFILE_DUPLICATE_KEY")
        result[key] = value
    return result


def _no_number(_value):
    raise LoaderProfileError("PROFILE_NUMBER_UNSUPPORTED")


@dataclass(frozen=True, slots=True)
class BaseBinding:
    repo_id: str
    revision: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class TokenizerBinding:
    repo_id: str
    revision: str
    asset_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class LoaderSelection:
    backend: str
    factory: str


@dataclass(frozen=True, slots=True)
class ExpectedModel:
    class_name: str
    model_type: str
    view: str


@dataclass(frozen=True, slots=True)
class Numerics:
    quantization: str
    quant_type: str
    compute_dtype: str
    storage_dtype: str
    double_quant: bool


@dataclass(frozen=True, slots=True)
class Placement:
    device: str
    offload: str


@dataclass(frozen=True, slots=True)
class LoaderProfileObservation:
    """Declared expectations matched; no actual load or authorization observed."""

    profile_sha256: str
    candidate_sha256: str
    recipe_sha256: str
    purpose: str
    base: BaseBinding
    tokenizer: TokenizerBinding
    runtime_identity_sha256: str
    loader: LoaderSelection
    expected_model: ExpectedModel
    numerics: Numerics
    placement: Placement

    def __bool__(self):
        raise TypeError("LOADER_PROFILE_OBSERVATION_IS_NOT_AUTHORIZATION")

    @property
    def evidence_class(self):
        return "MODELED"

    @property
    def state(self):
        return "BLOCKED"

    @property
    def signing_trust(self):
        return "UNKNOWN"

    @property
    def execution_authority(self):
        return False

    @property
    def training_admitted(self):
        return False

    @property
    def publication_eligible(self):
        return False


def observe_loader_profile(raw: bytes, *, expected_raw_sha256: str,
                           expected_candidate_sha256: str, expected_recipe_sha256: str,
                           expected_purpose: str) -> LoaderProfileObservation:
    """Compare exact descriptor bytes and externally supplied snapshot bindings.

    Expected arguments are comparison inputs, not authenticated owner pins.
    Errors are value-free. No loader is selected or called at runtime here.
    """
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_PROFILE_BYTES, "PROFILE_BYTES_INVALID")
    for expected in (expected_raw_sha256, expected_candidate_sha256, expected_recipe_sha256):
        _matches(expected, _HEX64, "EXPECTED_DIGEST_INVALID")
    _choice(expected_purpose, _PURPOSES, "EXPECTED_PURPOSE_INVALID")
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == expected_raw_sha256, "PROFILE_DIGEST_MISMATCH")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                           parse_float=_no_number, parse_int=_no_number, parse_constant=_no_number)
    except LoaderProfileError:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise LoaderProfileError("PROFILE_JSON_INVALID") from None

    _fields(value, {"schema", "purpose", "candidate_sha256", "recipe_sha256", "base", "tokenizer",
                    "runtime_identity_sha256", "loader", "expected_model", "numerics", "placement"},
            "PROFILE_FIELDS_INVALID")
    _require(type(value["schema"]) is str and value["schema"] == SCHEMA, "PROFILE_SCHEMA_INVALID")
    _choice(value["purpose"], _PURPOSES, "PROFILE_PURPOSE_INVALID")
    for name in ("candidate_sha256", "recipe_sha256", "runtime_identity_sha256"):
        _matches(value[name], _HEX64, "PROFILE_BINDING_INVALID")
    _require(value["candidate_sha256"] == expected_candidate_sha256
             and value["recipe_sha256"] == expected_recipe_sha256
             and value["purpose"] == expected_purpose, "PROFILE_BINDING_MISMATCH")

    base = value["base"]
    _fields(base, {"repo_id", "revision", "config_sha256"}, "BASE_FIELDS_INVALID")
    _matches(base["repo_id"], _REPO, "BASE_REPOSITORY_INVALID")
    _matches(base["revision"], _HEX40, "BASE_REVISION_INVALID")
    _matches(base["config_sha256"], _HEX64, "BASE_CONFIG_DIGEST_INVALID")
    tokenizer = value["tokenizer"]
    _fields(tokenizer, {"repo_id", "revision", "asset_manifest_sha256"}, "TOKENIZER_FIELDS_INVALID")
    _matches(tokenizer["repo_id"], _REPO, "TOKENIZER_REPOSITORY_INVALID")
    _matches(tokenizer["revision"], _HEX40, "TOKENIZER_REVISION_INVALID")
    _matches(tokenizer["asset_manifest_sha256"], _HEX64, "TOKENIZER_ASSET_DIGEST_INVALID")

    loader = value["loader"]
    _fields(loader, {"backend", "factory"}, "LOADER_FIELDS_INVALID")
    _require(type(loader["backend"]) is str and type(loader["factory"]) is str
             and (loader["backend"], loader["factory"]) in _LOADER_PAIRS, "LOADER_PAIR_UNSUPPORTED")
    model = value["expected_model"]
    _fields(model, {"class_name", "model_type", "view"}, "MODEL_FIELDS_INVALID")
    _require(type(model["class_name"]) is str and len(model["class_name"]) <= 256, "MODEL_CLASS_INVALID")
    _matches(model["class_name"], _CLASS, "MODEL_CLASS_INVALID")
    _matches(model["model_type"], _MODEL_TYPE, "MODEL_TYPE_INVALID")
    _choice(model["view"], {"full_model", "text_submodel"}, "MODEL_VIEW_UNSUPPORTED")

    numerics = value["numerics"]
    _fields(numerics, {"quantization", "quant_type", "compute_dtype", "storage_dtype", "double_quant"},
            "NUMERICS_FIELDS_INVALID")
    _choice(numerics["quantization"], {"none", "bitsandbytes_4bit"}, "QUANTIZATION_UNSUPPORTED")
    _choice(numerics["compute_dtype"], _FLOAT_DTYPES, "COMPUTE_DTYPE_UNSUPPORTED")
    _require(type(numerics["double_quant"]) is bool, "DOUBLE_QUANT_INVALID")
    if numerics["quantization"] == "none":
        _choice(numerics["storage_dtype"], _FLOAT_DTYPES, "STORAGE_DTYPE_UNSUPPORTED")
        _require(type(numerics["quant_type"]) is str and numerics["quant_type"] == "none"
                 and numerics["double_quant"] is False, "UNQUANTIZED_SETTINGS_CONFLICT")
    else:
        _choice(numerics["quant_type"], {"nf4", "fp4"}, "QUANT_TYPE_UNSUPPORTED")
        _require(type(numerics["storage_dtype"]) is str and numerics["storage_dtype"] == "uint8",
                 "QUANTIZED_STORAGE_UNSUPPORTED")
    placement = value["placement"]
    _fields(placement, {"device", "offload"}, "PLACEMENT_FIELDS_INVALID")
    _choice(placement["device"], {"cpu", "cuda"}, "DEVICE_UNSUPPORTED")
    _choice(placement["offload"], {"none"}, "OFFLOAD_UNSUPPORTED")

    return LoaderProfileObservation(
        profile_sha256=digest, candidate_sha256=value["candidate_sha256"],
        recipe_sha256=value["recipe_sha256"], purpose=value["purpose"],
        base=BaseBinding(**base), tokenizer=TokenizerBinding(**tokenizer),
        runtime_identity_sha256=value["runtime_identity_sha256"], loader=LoaderSelection(**loader),
        expected_model=ExpectedModel(**model), numerics=Numerics(**numerics), placement=Placement(**placement),
    )
