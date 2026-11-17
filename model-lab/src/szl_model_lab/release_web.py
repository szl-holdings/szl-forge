"""Authenticated read-only HTML and JSON release-identity comparison."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .release_decision import UnknownRail, load_decision, unavailable_decision


def register_release_decision(app, authorize, templates, settings) -> None:
    def state(invalidate):
        if settings.release_snapshot is None:
            if invalidate is not None:
                raise HTTPException(404, "unknown_release_rail")
            return unavailable_decision()
        try:
            return load_decision(
                settings.release_snapshot,
                settings.release_snapshot_sha256,
                invalidate=invalidate,
            )
        except UnknownRail:
            raise HTTPException(404, "unknown_release_rail") from None
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            return unavailable_decision("INVALID_OR_UNAVAILABLE_SNAPSHOT")

    @app.get("/api/release-decision", dependencies=[Depends(authorize)])
    def api_release_decision(
        invalidate: str | None = Query(None, max_length=16),
    ) -> JSONResponse:
        report = state(invalidate)
        return JSONResponse(
            report, status_code=200 if report["counts"] is not None else 503
        )

    @app.get(
        "/release-decision",
        response_class=HTMLResponse,
        dependencies=[Depends(authorize)],
    )
    def release_decision(
        invalidate: str | None = Query(None, max_length=16),
    ) -> HTMLResponse:
        report = state(invalidate)
        content = templates.get_template("release-decision.html").render(
            decision=report
        )
        return HTMLResponse(
            content, status_code=200 if report["counts"] is not None else 503
        )
