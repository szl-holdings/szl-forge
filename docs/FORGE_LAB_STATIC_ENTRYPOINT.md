# Forge Lab static entry-point verification

The existing Forge Lab publisher verifies the exact protected Git source,
immutable Hub revision, and published file bytes before live smoke checks.
Its static host may serve `GET /` as HTTP 302 with the exact relative
`Location: /index.html`; the HTML entry point is served by `GET /index.html`.

The smoke contract accepts that root response only in static mode on the
canonical HTTPS `*.static.hf.space` host, and only when `/index.html` is also
an independently required smoke probe. That target must return a nonempty
HTTP 200 within the original shared publication deadline. The receipt retains
the root's actual 302 and names the separately verified target probe.

Every request keeps `allow_redirects=False`. Absolute or cross-origin
locations, altered paths, query strings or fragments, other redirect statuses,
redirects from other paths, and all dynamic-Space redirects remain failures.
A missing, empty, redirected, unavailable, or late entry-point response cannot
complete verification. No retry budget or source/byte qualification changes.

The existing workflow continues checking the portfolio and frozen Foundation
showcase. Its push paths include the shared publisher and its controls, so a
merged verifier repair enters this established publication route. This source
contract does not itself establish live publication or model qualification;
the canonical workflow and immutable readback must provide that evidence.
