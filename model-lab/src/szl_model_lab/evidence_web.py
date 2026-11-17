"""Register evidence HTML and JSON against one shared, read-only projector."""
from __future__ import annotations
from urllib.parse import urlencode

from fastapi import Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .evidence_view import SelectionError, load_lineage, unavailable_lineage


def register_evidence(app, authorize, templates, settings) -> None:
    def state(model, invalidate):
        if settings.evidence_snapshot is None:
            return unavailable_lineage()
        try:
            return load_lineage(settings.evidence_snapshot, settings.evidence_snapshot_sha256,
                                model=model, invalidate=invalidate)
        except SelectionError:
            raise HTTPException(404, 'unknown_evidence_selection') from None
        except (ValueError, OSError, TypeError, KeyError, OverflowError):
            return unavailable_lineage('INVALID_OR_UNAVAILABLE_SNAPSHOT')

    @app.get('/api/evidence-lineage', dependencies=[Depends(authorize)])
    def api_lineage(model: str | None = Query(None, max_length=120),
                    invalidate: str | None = Query(None, max_length=64)) -> JSONResponse:
        data = state(model, invalidate)
        return JSONResponse(data, status_code=200 if data['counts'] is not None else 503)

    @app.get('/evidence', response_class=HTMLResponse, dependencies=[Depends(authorize)])
    def evidence_page(model: str | None = Query(None, max_length=120),
                      invalidate: str | None = Query(None, max_length=64)) -> HTMLResponse:
        data = state(model, invalidate)
        query = {k: v for k, v in {'model': model, 'invalidate': invalidate}.items() if v is not None}
        body = templates.get_template('evidence.html').render(
            evidence=data, api_url='/api/evidence-lineage?' + urlencode(query),
            baseline_url='/evidence?' + urlencode({'model': data['selected_model'] or ''}),
            node_labels={node['node_id']: node['label'] for node in data['nodes']})
        return HTMLResponse(body, status_code=200 if data['counts'] is not None else 503)
