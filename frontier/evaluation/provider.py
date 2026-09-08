# SPDX-License-Identifier: Apache-2.0
"""Provider selection and measured failure-to-baseline fallback."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .core import canonical_sha256, execute_case

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
BASELINE_URL = "https://szlholdings-szl-model-inference-lab.hf.space"
PROVIDERS = (
    "baseten",
    "zai-org",
    "deepinfra",
    "fireworks-ai",
    "novita",
    "together",
)


class ProviderError(RuntimeError):
    pass


def choose_provider(
    first_case: Mapping[str, Any],
    *,
    model_id: str,
    token: str,
    providers: Sequence[str],
    timeout: float,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for provider in providers:
        routed_model = f"{model_id}:{provider}"
        result = execute_case(
            first_case,
            route="candidate",
            model=routed_model,
            url=ROUTER_URL,
            token=token,
            max_tokens=96,
            timeout=timeout,
        )
        attempts.append(
            {
                "provider": provider,
                "model": routed_model,
                "ok": result["ok"],
                "http_status": result.get("http_status"),
                "latency_ms": result.get("latency_ms"),
                "error_type": result.get("error_type"),
                "error_sha256": result.get("error_sha256"),
            }
        )
        if result["ok"]:
            return provider, result, attempts
    raise ProviderError(
        "no configured provider completed the first fixture; attempts_sha256="
        + canonical_sha256(attempts)
    )


def exercise_fallback(
    case: Mapping[str, Any],
    *,
    model_id: str,
    provider: str,
    token: str,
    baseline_model: str,
    timeout: float,
) -> dict[str, Any]:
    failed = execute_case(
        case,
        route="intentional-provider-failure",
        model=f"{model_id}-szl-intentional-invalid:{provider}",
        url=ROUTER_URL,
        token=token,
        max_tokens=8,
        timeout=min(timeout, 45.0),
    )
    fallback = execute_case(
        case,
        route="baseline-fallback",
        model=baseline_model,
        url=f"{BASELINE_URL}/v1/chat/completions",
        token=None,
        max_tokens=32,
        timeout=timeout,
    )
    return {
        "invalid_candidate_request_failed": not failed["ok"],
        "invalid_candidate_http_status": failed.get("http_status"),
        "invalid_candidate_error_type": failed.get("error_type"),
        "invalid_candidate_error_sha256": failed.get("error_sha256"),
        "baseline_fallback_succeeded": bool(fallback["ok"]),
        "baseline_fallback_score": fallback.get("score"),
        "baseline_fallback_latency_ms": fallback.get("latency_ms"),
        "pass": (not failed["ok"]) and bool(fallback["ok"]),
    }
