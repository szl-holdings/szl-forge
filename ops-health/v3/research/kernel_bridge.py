"""Load the v3 kernel (v3/ops-health/ops_health.py) as a module for research code.

The kernel's arithmetic (screening, normalization with the transform table, linear predictor,
sigmoid, calibrator, nonconformity, prediction sets) is the single source of truth: research
code scores rows with these functions. The directory name contains a hyphen, so the file is
loaded by path. ``load(path)`` also loads temporary kernel copies (mutants, fail-open fixture)
under distinct module names.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from . import registry

_MODULE_NAME = "oac_ops_health_kernel_v3"
_CACHE: dict[str, ModuleType] = {}


def load(path: Path | str = registry.KERNEL_PATH) -> ModuleType:
    path = Path(path).resolve()
    key = str(path)
    if key not in _CACHE:
        name = _MODULE_NAME if path == registry.KERNEL_PATH.resolve() else f"{_MODULE_NAME}_{len(_CACHE)}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load kernel from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _CACHE[key] = module
    return _CACHE[key]


def kernel() -> ModuleType:
    return load()
