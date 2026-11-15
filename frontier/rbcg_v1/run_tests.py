#!/usr/bin/env python3
"""run_tests.py -- run the RBCG v1 suite. Exit 0 only if every test passed and none was skipped.
RBCG_FAILFAST=1 stops at the first failure (used by the mutation harness)."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in ("test_rbcg1", "test_rbcg1_log", "test_rbcg1_verify"):
        suite.addTests(loader.loadTestsFromName(name))
    res = unittest.TextTestRunner(verbosity=int(os.environ.get("RBCG_TEST_VERBOSITY", "1")),
                                  failfast=os.environ.get("RBCG_FAILFAST") == "1").run(suite)
    print(f"RBCG_SUITE ran={res.testsRun} failures={len(res.failures)} errors={len(res.errors)} "
          f"skipped={len(res.skipped)}", flush=True)
    return 0 if res.wasSuccessful() and not res.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
