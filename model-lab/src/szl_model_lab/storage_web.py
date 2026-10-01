# SPDX-License-Identifier: Apache-2.0
"""Authenticated calculation view in the existing Model Lab, with no I/O controls."""
# No postponed annotations: Request is resolved in FastAPI's actual route scope.
import re

from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .storage import GIB, reference_plan


def _query(request: Request) -> tuple[int, int]:
    values = request.query_params.multi_items()
    if len(values) > 2 or len({k for k, _ in values}) != len(values):
        raise HTTPException(422, "duplicate_or_excess_plan_parameters")
    parsed = {"tokens": 32768, "batch": 1}
    for key, value in values:
        if key not in parsed or re.fullmatch(r"[1-9][0-9]{0,6}", value) is None:
            raise HTTPException(422, "invalid_plan_parameter")
        parsed[key] = int(value)
    if not 1 <= parsed["tokens"] <= 1048576 or not 1 <= parsed["batch"] <= 128:
        raise HTTPException(422, "plan_parameter_outside_bounds")
    return parsed["tokens"], parsed["batch"]


def register_storage(app, authorize, templates) -> None:
    """Reuse the parent's authentication, headers, packaging and route ownership."""
    paths = {"/storage", "/api/storage/plan"}
    if any(getattr(route, "path", None) in paths for route in app.routes):
        raise ValueError("storage_route_collision")

    @app.get("/api/storage/plan", dependencies=[Depends(authorize)])
    def storage_plan(request: Request) -> JSONResponse:
        tokens, batch = _query(request)
        return JSONResponse(reference_plan(tokens, batch))

    @app.get("/storage", response_class=HTMLResponse, dependencies=[Depends(authorize)])
    def storage_page(request: Request) -> HTMLResponse:
        tokens, batch = _query(request)
        plan = reference_plan(tokens, batch)
        return HTMLResponse(templates.get_template("storage.html").render(
            plan=plan, full_gib=f"{plan['kv_full_allocation_bytes']/GIB:,.3f}",
            bounded_gib=f"{plan['kv_if_bounded_sliding_bytes']/GIB:,.3f}"))
