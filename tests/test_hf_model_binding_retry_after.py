from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import publish_model_source_bindings as bindings


class HubModelInfoRetryTests(unittest.TestCase):
    def test_honors_numeric_retry_after(self) -> None:
        class RateLimitError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("rate limited")
                self.response = SimpleNamespace(
                    status_code=429,
                    headers={"Retry-After": "17"},
                )

        expected = SimpleNamespace(sha="a" * 40, siblings=[])
        api = mock.Mock()
        api.model_info.side_effect = [RateLimitError(), expected]
        sleeps: list[float] = []

        observed = bindings._model_info_with_retry(
            api,
            "SZLHOLDINGS/example",
            sleeper=sleeps.append,
            max_attempts=3,
            total_sleep_budget_seconds=30,
            files_metadata=True,
        )

        self.assertIs(observed, expected)
        self.assertEqual([17.0], sleeps)
        self.assertEqual(2, api.model_info.call_count)

    def test_does_not_retry_nontransient_denial(self) -> None:
        class ProviderPermissionError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("denied")
                self.response = SimpleNamespace(status_code=403, headers={})

        api = mock.Mock()
        api.model_info.side_effect = ProviderPermissionError()
        sleeps: list[float] = []

        with self.assertRaises(ProviderPermissionError):
            bindings._model_info_with_retry(
                api,
                "SZLHOLDINGS/example",
                sleeper=sleeps.append,
                max_attempts=3,
            )

        self.assertEqual([], sleeps)
        self.assertEqual(1, api.model_info.call_count)

    def test_fails_closed_when_retry_budget_is_exhausted(self) -> None:
        class RateLimitError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("rate limited")
                self.response = SimpleNamespace(
                    status_code=429,
                    headers={"Retry-After": "181"},
                )

        api = mock.Mock()
        api.model_info.side_effect = RateLimitError()
        sleeps: list[float] = []

        with self.assertRaisesRegex(
            bindings.TransientBindingError,
            "retry budget exhausted",
        ):
            bindings._model_info_with_retry(
                api,
                "SZLHOLDINGS/example",
                sleeper=sleeps.append,
                max_attempts=3,
                total_sleep_budget_seconds=100,
            )

        self.assertEqual([], sleeps)
        self.assertEqual(1, api.model_info.call_count)

    def test_caps_numeric_retry_after(self) -> None:
        class RateLimitError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("rate limited")
                self.response = SimpleNamespace(
                    status_code=429,
                    headers={"retry-after": "999"},
                )

        expected = SimpleNamespace(sha="a" * 40, siblings=[])
        api = mock.Mock()
        api.model_info.side_effect = [RateLimitError(), expected]
        sleeps: list[float] = []

        bindings._model_info_with_retry(
            api,
            "SZLHOLDINGS/example",
            sleeper=sleeps.append,
            max_attempts=2,
            total_sleep_budget_seconds=180,
        )
        self.assertEqual([180.0], sleeps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
