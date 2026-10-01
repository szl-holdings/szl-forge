# Lyte metrics in the read-only mesh

`tools/szl_live_mesh.py` reads `/api/lyte/v2/metrics` as Prometheus text.
The JSON transport remains responsible for JSON endpoints. Previously, the
metrics URL entered the JSON-object loop and valid exporter text was rejected;
the positive fixture incorrectly supplied a JSON object instead of exposition.

The separate `MetricsResponse` transport retains bounded raw bytes, HTTP status,
Content-Type and Content-Encoding until validation. It accepts only the observed
`text/plain` format with explicit version `1.0.0` or `0.0.4`, `charset=utf-8`, identity encoding, strict
UTF-8 and at most 2,000,000 bytes. Parameter order, header case and quoted parameter
values are accepted. Missing, duplicate or unsupported format declarations fail.
Lyte's pinned `prometheus-client==0.26.0` exports `CONTENT_TYPE_LATEST` as version
1.0.0; 0.0.4 is the supported compatibility format. The observer negotiates both
and records the actual validated version without inferring it from body text.
Redirects, ambient proxies, credentials and automatic retries remain disabled.
The socket timeout does not replace the caller's process/job wall-clock limit.

The observer reads the publisher's unique `SOURCE_REVISION` and `EXPECTED_VERSION`
top-level literal assignments from the exact observed A11oy commit. Python AST
inspection rejects additional static writes without executing the source. It checks Lyte readiness before
reading metrics, then requires one `lyte_build_info` gauge with those labels and
value 1, and one unlabeled `lyte_db_pool_healthy` gauge with value 1. Gauge type
declarations, uniqueness, finite values and exact labels are required. This is a
critical-gauge contract; other metric families are not validated. A11oy and Lyte
source heads must be available and unchanged at both ends of the observation.

The critical-gauge parser and its constants are copied verbatim from
[`scripts/lyte_enterprise_live_contract.py`](https://github.com/szl-holdings/a11oy/blob/43058398fb8ea346a7bd977f1a35391aeec1bf1a/scripts/lyte_enterprise_live_contract.py)
under its Apache-2.0 license. The source Git blob is
`494116f65b24ea5b8a516abde31668be2248cbf4`; SHA-256 of the LF-normalized extracted
constants and function is
`52e62df293baa41edfd5a80af9bc4c9a65e72d276ef900e3a447280c08f8f5a3`.
Updates must be reconciled with that source owner. No remotely acquired Python
is executed by the observer.

`lyte_metrics` in the report contains status, byte count, response digest,
expected source/version, validated canonical headers and separate results for
gauges, readiness and source bookends. Response bodies and unrecognized header
values are not returned. A positive parser result alone does not establish parity.

Offline consumers injecting `fetch` must also inject `metrics_fetch`, returning
`MetricsResponse`. Omitting it fails closed without falling back to the network.
Existing JSON adapters retain their `(status, body)` contract.

Run the actual consumer regression and transport cases with:

```sh
python -m unittest discover -s tests -p test_szl_live_mesh.py -v
```

The fixtures are synthetic and perform no network requests. Linux is needed to
exercise the existing symlink cases when Windows lacks symlink privileges.
The native base Python gate also runs these tests with the complete `tests/`
suite on Python 3.11 and 3.12.

Every observation remains `state=HOLD`, `production_authorization=false` and
unsigned. Source and metrics consistency does not authenticate a deployment,
qualify a model, complete a browser journey or authorize publication.
