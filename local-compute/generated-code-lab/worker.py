"""Executed only inside the restricted Docker container, never on the host.
SPDX-License-Identifier: Apache-2.0
"""
import importlib.util
import json
import sys

values = json.loads(sys.stdin.read(65537))
spec = importlib.util.spec_from_file_location("candidate_solution", "/candidate/solution.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
answers = [module.solve(value) for value in values]
print(json.dumps(answers, ensure_ascii=False, allow_nan=False))
