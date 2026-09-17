"""Deterministic transport/clock tests; no HTTP, tokens, sleeps or model calls."""
from __future__ import annotations

import importlib.util
import io
import json
from datetime import datetime, timezone
from email.message import Message
from email.utils import format_datetime
from pathlib import Path
import sys
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rate_census", ROOT / "tools/observe_public_estate_files.py")
assert spec is not None and spec.loader is not None
census = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = census
spec.loader.exec_module(census)
HF = "https://huggingface.co/api/models?author=SZLHOLDINGS&limit=100"
GH = "https://api.github.com/orgs/szl-holdings/repos"
REVISION = "a" * 40
EPOCH = datetime(2026, 9, 16, tzinfo=timezone.utc).timestamp()


def headers(values=None):
    result = Message()
    for key, value in (values or {}).items():
        result[key] = value
    return result


def http_date(seconds):
    return format_datetime(datetime.fromtimestamp(EPOCH + seconds, tz=timezone.utc), usegmt=True)


class Clock:
    def __init__(self):
        self.value = 0.0
        self.waits = []
        self.on_wait = None

    def monotonic(self):
        return self.value

    def wall(self):
        return EPOCH + self.value

    def sleep(self, seconds):
        if self.on_wait is not None:
            self.on_wait(seconds)
        self.waits.append(seconds)
        self.value += seconds


class Reply:
    def __init__(self, payload=None, *, status=200, fields=None, raw=None):
        self.status = status
        self.headers = headers(fields)
        self.body = json.dumps([] if payload is None else payload).encode() if raw is None else raw
        self.closed = False
        self.reads = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self, limit):
        self.reads.append(limit)
        return self.body[:limit]


class ErrorBody(io.BytesIO):
    def read(self, *args):
        raise AssertionError("Error bodies must not be consumed")


def error(code=429, fields=None, url=HF):
    return HTTPError(url, code, "untrusted-error-marker", headers(fields), ErrorBody(b"secret-error-body"))


class RecordingOpener:
    def __init__(self, steps, clock):
        self.steps = list(steps)
        self.clock = clock
        self.calls = []

    def open(self, request, *, timeout):
        self.calls.append({"url": request.full_url, "method": request.method,
                           "headers": dict(request.header_items()), "time": self.clock.value,
                           "timeout": timeout})
        if not self.steps:
            raise AssertionError("Unexpected additional network attempt")
        result = self.steps.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def reader(steps, *, token=None):
    clock = Clock()
    client = census.Client(token, clock=clock.monotonic, sleeper=clock.sleep, wall_clock=clock.wall)
    client.opener = RecordingOpener(steps, clock)
    return client, clock


class HeaderHints(unittest.TestCase):
    def test_absent_header(self):
        self.assertEqual(census.hf_rate_hint(headers(), EPOCH)["state"], "ABSENT")

    def test_missing_header_container_is_absent_not_an_exception(self):
        self.assertEqual(census.hf_rate_hint(None, EPOCH)["state"], "ABSENT")

    def test_numeric_retry_after(self):
        hint = census.hf_rate_hint(headers({"Retry-After": "20"}), EPOCH)
        self.assertEqual(hint["retry_after_seconds"], 20)

    def test_http_date(self):
        hint = census.hf_rate_hint(headers({"Retry-After": http_date(20)}), EPOCH)
        self.assertEqual(hint["retry_after_seconds"], 20)

    def test_clock_skew_cannot_shorten_server_relative_wait(self):
        hint = census.hf_rate_hint(headers({"Retry-After": http_date(20), "Date": http_date(0)}), EPOCH + 15)
        self.assertEqual(hint["retry_after_seconds"], 20)

    def test_past_http_date_is_zero(self):
        hint = census.hf_rate_hint(headers({"Retry-After": http_date(-20)}), EPOCH)
        self.assertEqual(hint["retry_after_seconds"], 0)

    def test_rate_parameters_in_either_order(self):
        for raw in ('"api";r=4;t=20', '"api";t=20;r=4', '"api" ; r = 4 ; t = 20'):
            with self.subTest(raw=raw):
                hint = census.hf_rate_hint(headers({"RateLimit": raw}), EPOCH)
                self.assertEqual((hint["remaining"], hint["reset_seconds"]), (4, 20))

    def test_combined_bucket_and_multiple_policies(self):
        hint = census.hf_rate_hint(headers({"RateLimit": '"api|pages|resolvers";r=3;t=10, "api";r=1;t=20'}), EPOCH)
        self.assertEqual((hint["remaining"], hint["reset_seconds"]), (1, 20))

    def test_other_buckets_do_not_define_api_reset(self):
        hint = census.hf_rate_hint(headers({"RateLimit": '"resolvers";r=0;t=100'}), EPOCH)
        self.assertEqual(hint["state"], "ABSENT")

    def test_retry_after_invalid_values(self):
        for raw in ("", "-1", "+1", "1.5", "1e9", "inf", "nan", "9" * 40, "bad-date", "x\r\nX: y"):
            with self.subTest(raw=raw):
                self.assertEqual(census.hf_rate_hint({"Retry-After": raw}, EPOCH)["state"], "INVALID")

    def test_rate_limit_invalid_values(self):
        for raw in ('"api";r=0', '"api";r=0;t=1;t=2', '"api";r=-1;t=1',
                    '"api";r=0;t=1.5', 'api;r=0;t=1', '"api";r=0;t=1;evil=2',
                    '"api";r=0;t=1,', '"api";r=0;t=1\n', '"api";r=0;t=' + '1' * 11):
            with self.subTest(raw=raw):
                self.assertEqual(census.hf_rate_hint({"RateLimit": raw}, EPOCH)["state"], "INVALID")

    def test_header_bound_and_duplicate(self):
        for key in ("Retry-After", "RateLimit"):
            with self.subTest(key=key):
                self.assertEqual(census.hf_rate_hint({key: "a" * 1025}, EPOCH)["state"], "INVALID")
                duplicate = headers({key: "1"})
                duplicate[key] = "2"
                self.assertEqual(census.hf_rate_hint(duplicate, EPOCH)["state"], "INVALID")

    def test_invalid_date_hint_is_not_normalized(self):
        hint = census.hf_rate_hint(headers({"Retry-After": http_date(20), "Date": "not-date"}), EPOCH)
        self.assertEqual(hint["state"], "INVALID")

    def test_header_case_insensitive(self):
        hint = census.hf_rate_hint(headers({"retry-after": "2", "ratelimit": '"api";r=0;t=3'}), EPOCH)
        self.assertEqual(hint["retry_after_seconds"], 2)
        self.assertEqual(hint["reset_seconds"], 3)

    def test_nonstring_header_is_invalid(self):
        for value in (1, True, [], {}):
            with self.subTest(value=value):
                self.assertEqual(census.hf_rate_hint({"Retry-After": value}, EPOCH)["state"], "INVALID")


class Backpressure(unittest.TestCase):
    def test_serial_hf_launches_are_paced(self):
        c, clock = reader([Reply(), Reply()])
        c.get(HF); c.get(HF)
        self.assertEqual([row["time"] for row in c.opener.calls], [0, 1])
        self.assertEqual(clock.waits, [1])

    def test_github_is_not_delayed_by_hf_pacing(self):
        c, clock = reader([Reply(), Reply(), Reply()])
        c.get(HF); c.get(GH); c.get(GH)
        self.assertEqual(clock.waits, [])

    def test_github_429_remains_a_failure_not_hf_retry(self):
        failure = error(url=GH)
        c, clock = reader([failure])
        with self.assertRaisesRegex(census.CensusError, "HTTP_429"):
            c.get(GH)
        self.assertTrue(failure.fp.closed)
        self.assertEqual(c.calls, 1)
        self.assertEqual(clock.waits, [])

    def test_retry_after_then_success(self):
        c, clock = reader([error(fields={"Retry-After": "7"}), Reply([1])])
        value, _ = c.get(HF)
        self.assertEqual(value, [1])
        self.assertEqual(clock.waits, [7])
        self.assertEqual([r["http_status"] for r in c.responses], [429, 200])

    def test_rate_limit_reset_then_success(self):
        c, clock = reader([error(fields={"RateLimit": '"api";r=0;t=13'}), Reply()])
        c.get(HF)
        self.assertEqual(clock.waits, [13])

    def test_both_hints_honor_longer_wait(self):
        c, clock = reader([error(fields={"Retry-After": "7", "RateLimit": '"api";r=0;t=13'}), Reply()])
        c.get(HF)
        self.assertEqual(clock.waits, [13])

    def test_http_date_retry(self):
        c, clock = reader([error(fields={"Retry-After": http_date(20)}), Reply()])
        c.get(HF)
        self.assertEqual(clock.waits, [20])

    def test_no_hint_has_bounded_conservative_wait(self):
        c, clock = reader([error(), Reply()])
        c.get(HF)
        self.assertEqual(clock.waits, [300])

    def test_zero_wait_cannot_create_busy_loop(self):
        c, clock = reader([error(fields={"Retry-After": "0"}), Reply()])
        c.get(HF)
        self.assertEqual(clock.waits, [1])

    def test_success_quota_is_spread_over_reset(self):
        c, clock = reader([Reply(fields={"RateLimit": '"api";r=2;t=20'}), Reply()])
        c.get(HF); c.get(HF)
        self.assertEqual(clock.waits, [10])

    def test_success_exhausted_quota_waits_for_reset(self):
        c, clock = reader([Reply(fields={"RateLimit": '"api";r=0;t=20'}), Reply()])
        c.get(HF); c.get(HF)
        self.assertEqual(clock.waits, [20])

    def test_error_response_closed_before_sleep_and_never_read(self):
        failure = error(fields={"Retry-After": "2"})
        c, clock = reader([failure, Reply()])
        clock.on_wait = lambda _: self.assertTrue(failure.fp.closed)
        c.get(HF)
        self.assertNotIn("body_bytes", c.responses[0])
        self.assertIsNone(c.responses[0]["body_sha256"])

    def test_status_429_response_is_also_closed_before_sleep(self):
        reply = Reply(status=429, fields={"Retry-After": "3"})
        c, clock = reader([reply, Reply()])
        clock.on_wait = lambda _: self.assertTrue(reply.closed)
        c.get(HF)
        self.assertEqual(reply.reads, [])
        self.assertEqual(clock.waits, [3])

    def test_invalid_hint_defers_without_guessing(self):
        c, clock = reader([error(fields={"Retry-After": "nonsense"})])
        with self.assertRaisesRegex(census.CensusError, "HF_RATE_HINT_INVALID"):
            c.get(HF)
        self.assertEqual(clock.waits, [])
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.responses[0]["hf_retry"]["decision"], "DEFER")

    def test_long_hint_is_never_shortened_to_budget(self):
        c, clock = reader([error(fields={"RateLimit": '"api";r=0;t=601'})])
        with self.assertRaisesRegex(census.CensusError, "HF_RATE_WAIT_BUDGET"):
            c.get(HF)
        self.assertEqual(clock.waits, [])
        self.assertEqual(c.responses[0]["hf_retry"]["delay_seconds"], 601)

    def test_per_url_attempt_cap_stops_later_hf_families(self):
        c, clock = reader([error(fields={"Retry-After": "1"}) for _ in range(3)])
        with self.assertRaisesRegex(census.CensusError, "HF_RATE_RETRY_BOUND"):
            c.get(HF)
        for family in ("models", "datasets", "spaces", "kernels"):
            with self.assertRaisesRegex(census.CensusError, "HF_RATE_RETRY_BOUND"):
                c.get(HF.replace("models", family))
        self.assertEqual(c.calls, 3)
        self.assertEqual(c.rate_limit_receipt()["deferred_calls_without_network"], 4)
        self.assertEqual(clock.waits, [1, 1])

    def test_total_retry_cap_applies_across_successful_gets(self):
        steps = []
        for _ in range(3):
            steps += [error(fields={"Retry-After": "1"}), error(fields={"Retry-After": "1"}), Reply()]
        steps.append(error(fields={"Retry-After": "1"}))
        c, _ = reader(steps)
        for _ in range(3): c.get(HF)
        with self.assertRaisesRegex(census.CensusError, "HF_RATE_RETRY_BOUND"):
            c.get(HF)
        self.assertEqual(c.calls, 10)
        self.assertEqual(c.rate_limit_receipt()["retries_scheduled"], 6)

    def test_cumulative_wait_cap_across_gets(self):
        c, _ = reader([error(), Reply(), error(), Reply(), error(fields={"Retry-After": "1"})])
        c.get(HF); c.get(HF)
        with self.assertRaisesRegex(census.CensusError, "HF_RATE_WAIT_BUDGET"):
            c.get(HF)
        self.assertEqual(c.rate_limit_receipt()["retry_wait_seconds_reserved"], 600)

    def test_retry_that_would_exceed_launch_deadline_is_deferred(self):
        c, clock = reader([error(fields={"Retry-After": "20"})])
        clock.value = 890
        with self.assertRaisesRegex(census.CensusError, "HF_WAIT_EXCEEDS_LAUNCH_DEADLINE"):
            c.get(HF)
        self.assertEqual(clock.waits, [])
        self.assertEqual(c.calls, 1)

    def test_success_long_reset_prevents_next_launch(self):
        c, clock = reader([Reply(fields={"RateLimit": '"api";r=0;t=901'})])
        c.get(HF)
        with self.assertRaisesRegex(census.CensusError, "HF_WAIT_EXCEEDS_LAUNCH_DEADLINE"):
            c.get(HF)
        self.assertEqual(c.calls, 1)
        self.assertEqual(clock.waits, [])

    def test_request_cap_counts_retries_and_checks_before_wait(self):
        c, clock = reader([error(fields={"Retry-After": "2"})])
        c.calls = census.MAX_REQUESTS - 1
        with self.assertRaisesRegex(census.CensusError, "REQUEST_BOUND"):
            c.get(HF)
        self.assertEqual(clock.waits, [])
        self.assertEqual(len(c.opener.calls), 1)

    def test_deadline_rechecked_after_slow_sleep(self):
        c, clock = reader([Reply()])
        c.get(HF)
        clock.on_wait = lambda _: setattr(clock, "value", 901)
        with self.assertRaisesRegex(census.CensusError, "REQUEST_LAUNCH_DEADLINE"):
            c.get(HF)
        self.assertEqual(c.calls, 1)

    def test_byte_cap_rechecked_after_wait(self):
        c, clock = reader([Reply()])
        c.get(HF)
        clock.on_wait = lambda _: setattr(c, "total_bytes", census.MAX_TOTAL_BYTES)
        with self.assertRaisesRegex(census.CensusError, "TOTAL_BYTE_BOUND"):
            c.get(HF)
        self.assertEqual(c.calls, 1)

    def test_nonadvancing_clock_stops_without_retry_spin(self):
        c, _ = reader([Reply()])
        c._sleep = lambda _: None
        c.get(HF)
        with self.assertRaisesRegex(census.CensusError, "HF_WAIT_CLOCK_NOT_ADVANCING"):
            c.get(HF)
        self.assertEqual(c.calls, 1)

    def test_repeated_short_sleep_has_finite_progress_guard(self):
        c, clock = reader([Reply()])
        c._sleep = lambda _: setattr(clock, "value", clock.value + 0.01)
        c.get(HF)
        with self.assertRaisesRegex(census.CensusError, "HF_WAIT_CLOCK_PROGRESS_BOUND"):
            c.get(HF)
        self.assertEqual(c.calls, 1)

    def test_keyboard_interrupt_stops_no_second_attempt(self):
        failure = error(fields={"Retry-After": "2"})
        c, _ = reader([failure])
        c._sleep = mock.Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt): c.get(HF)
        self.assertEqual(c.calls, 1)
        self.assertTrue(failure.fp.closed)

    def test_non429_http_errors_not_retried(self):
        for code in (301, 302, 401, 403, 404, 500, 503):
            with self.subTest(code=code):
                failure = error(code)
                c, clock = reader([failure])
                with self.assertRaisesRegex(census.CensusError, f"HTTP_{code}"):
                    c.get(HF)
                self.assertEqual(c.calls, 1)
                self.assertEqual(clock.waits, [])
                self.assertTrue(failure.fp.closed)

    def test_transport_error_not_retried(self):
        c, _ = reader([URLError("sensitive-text")])
        with self.assertRaisesRegex(census.CensusError, "^TRANSPORT_UNAVAILABLE$"):
            c.get(HF)
        self.assertEqual(c.calls, 1)

    def test_retry_preserves_exact_url_get_and_no_hf_credentials(self):
        c, _ = reader([error(fields={"Retry-After": "2"}), Reply()], token="fixture-only-github-token")
        c.get(HF)
        for attempt in c.opener.calls:
            self.assertEqual(attempt["method"], "GET")
            self.assertEqual(attempt["url"], HF)
            self.assertNotIn("authorization", {k.lower() for k in attempt["headers"]})
            self.assertEqual(attempt["timeout"], 12)

    def test_hf_circuit_does_not_hide_unrelated_github_read(self):
        c, _ = reader([error(fields={"Retry-After": "601"}), Reply([1])])
        with self.assertRaises(census.CensusError): c.get(HF)
        self.assertEqual(c.get(GH)[0], [1])

    def test_raw_headers_errors_and_credentials_are_not_in_receipts(self):
        c, _ = reader([error(fields={"Retry-After": "SECRET-HEADER-MARKER"})], token="fixture-only-github-token")
        with self.assertRaises(census.CensusError): c.get(HF)
        encoded = json.dumps({"responses": c.responses, "control": c.rate_limit_receipt()})
        for value in ("SECRET-HEADER-MARKER", "secret-error-body", "untrusted-error-marker", "fixture-only-github-token"):
            self.assertNotIn(value, encoded)

    def test_returned_link_preserved_after_retry(self):
        link = f'<{HF}&cursor=x>; rel="next"'
        c, _ = reader([error(fields={"Retry-After": "1"}), Reply(fields={"Link": link})])
        self.assertEqual(c.get(HF)[1], link)

    def test_body_byte_limit_still_enforced(self):
        c, _ = reader([Reply(fields={"Content-Length": str(census.MAX_BODY + 1)})])
        with self.assertRaisesRegex(census.CensusError, "BODY_BOUND"): c.get(HF)
        self.assertEqual(c.calls, 1)

    def test_duplicate_json_remains_nonretryable(self):
        c, _ = reader([Reply(raw=b'{"a":1,"a":2}')])
        with self.assertRaises(census.CensusError): c.get(HF)
        self.assertEqual(c.calls, 1)

    def test_oversized_link_remains_nonretryable(self):
        c, _ = reader([Reply(fields={"Link": "x" * 8193})])
        with self.assertRaisesRegex(census.CensusError, "LINK_BOUND"): c.get(HF)
        self.assertEqual(c.calls, 1)

    def test_compressed_body_remains_nonretryable(self):
        c, _ = reader([Reply(fields={"Content-Encoding": "gzip"})])
        with self.assertRaisesRegex(census.CensusError, "COMPRESSED_RESPONSE_REFUSED"): c.get(HF)
        self.assertEqual(c.calls, 1)


class CensusIntegration(unittest.TestCase):
    def test_recovered_get_still_requires_ref_readback(self):
        info = {"id": "SZLHOLDINGS/a", "sha": REVISION, "private": False}
        entries = [{"path": "README.md", "type": "file", "oid": "b" * 40, "size": 1}]
        c, _ = reader([Reply(info), error(fields={"Retry-After": "1"}), Reply(entries), Reply(info)])
        row = census.observe_hf_repo(c, "models", "SZLHOLDINGS/a")
        self.assertTrue(row["complete"])
        self.assertEqual(row["revision_after"], REVISION)
        self.assertEqual(c.calls, 4)
        self.assertFalse(row["runtime_verified"])
        self.assertFalse(row["content_bytes_verified"])

    def test_ref_movement_after_recovery_remains_incomplete(self):
        info = {"id": "SZLHOLDINGS/a", "sha": REVISION, "private": False}
        c, _ = reader([Reply(info), error(fields={"Retry-After": "1"}), Reply([]), Reply(dict(info, sha="b" * 40))])
        row = census.observe_hf_repo(c, "models", "SZLHOLDINGS/a")
        self.assertFalse(row["complete"])
        self.assertIn("REF_MOVED", row["blockers"])

    def test_circuit_returns_partial_report_not_empty_estate(self):
        c, _ = reader([error(fields={"Retry-After": "1"}) for _ in range(3)])
        report = census.run(c, "huggingface", REVISION)
        self.assertEqual(report["status"], "PARTIAL_OR_UNAVAILABLE")
        self.assertFalse(report["complete"])
        self.assertEqual(report["requests_attempted"], 3)
        self.assertEqual(report["hf_rate_control"]["deferred_calls_without_network"], 3)
        for pop in report["populations"].values():
            self.assertFalse(pop["complete"])
            self.assertIsNone(pop["complete_scope_file_count"])
            self.assertIn("HF_RATE_RETRY_BOUND", pop["blockers"])
        for key in ("runtime_verified", "production_authorization", "semantic_review_complete"):
            self.assertFalse(report[key])
        self.assertEqual(report["source_content_files_read"], 0)

    def test_census_does_not_replay_historical_successes(self):
        c, _ = reader([error(fields={"Retry-After": "601"})])
        report = census.run(c, "huggingface", REVISION)
        self.assertEqual(len(report["responses"]), 1)
        self.assertFalse(report["hf_rate_control"]["resume_across_runs"])
        self.assertFalse(report["hf_rate_control"]["provider_quota_guaranteed"])

    def test_this_change_does_not_relax_lfs_identity(self):
        with self.assertRaisesRegex(census.CensusError, "HF_LFS_IDENTITY"):
            census.hf_file_rows([{"path": "model.safetensors", "type": "file", "oid": REVISION,
                                  "size": 1, "lfs": {"oid": "z" * 64}}])

    def test_response_accounting_includes_every_429_attempt(self):
        c, _ = reader([error(fields={"Retry-After": "2"}), error(fields={"Retry-After": "3"}), Reply()])
        c.get(HF)
        self.assertEqual(c.calls, len(c.responses))
        self.assertEqual([r["attempt"] for r in c.responses], [1, 2, 3])
        self.assertEqual([r["prelaunch_wait_seconds"] for r in c.responses], [0, 2, 3])
        self.assertEqual([r["http_status"] for r in c.responses], [429, 429, 200])
        self.assertEqual(c.total_bytes, len(b"[]"))


if __name__ == "__main__":
    unittest.main()
