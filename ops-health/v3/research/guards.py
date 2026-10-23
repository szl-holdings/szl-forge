"""Guards of OAC Ops Health v3 (SYNTHETIC only; protocol s.3.2 item 5, s.3.3, s.15.5).

1. **v2 seal guard** (s.3.2 item 5). The v3 generator refuses namespace ``oac-ops-health-v2`` with
   any v2 sealed split name (``test`` or ``shift_*``). Names are compared after folding (NFKC,
   format characters removed, casefolded, whitespace and hyphen runs to ``_``, edges stripped),
   so ``" Test"``, ``"SHIFT-clock-skew"`` or a zero-width variant is refused too. Any other
   namespace than the v3 and the v2 one is refused outright.
2. **Sealed-path guard** (s.3.2 item 5). Every v3 data path refuses paths under
   ``v2/data/sealed/``: the check runs on the absolute path and on the fully resolved path
   (symbolic links followed), compared with ``os.path.normcase`` so a case variant on a
   case-insensitive file system is refused as well, and before anything is read. As in v2's
   ``dataio``, any path with a component named ``sealed`` (casefolded) is refused too.
3. **DEV rule** (s.15.5). DEV rows exist only under the split name ``dev``; at most 2,048 rows
   per regime per DEV run; no fit on more than 2,048 DEV rows; no P or H statistic, tolerance,
   detection ratio, power or WIN-chance quantity on DEV data. A DEV run is one process: the
   process-wide ``DEV_LEDGER`` counts the rows written per regime. Every DEV stream is a prefix
   of one fixed stream per (namespace, master seed, regime, mode), so the ledger keeps the
   longest prefix per stream and sums the streams of a regime. A paired draw (D mode, twin mode)
   writes two rows per draw and counts both. The D and no-D arms count as regime ``nominal``.
4. **Runtime pins** (s.13 step 9: "CPython 3.12.10, run as ``python -I`` with
   ``PYTHONHASHSEED=0``"). ``runtime_pin_status`` reports them; ``require_runtime_pins`` refuses a
   registered computation that does not run under them. The check is the registered text, item
   by item: the interpreter is CPython 3.12.10, it runs isolated (``sys.flags.isolated``, set by
   ``-I``), and the process environment holds ``PYTHONHASHSEED=0``. MEASURED on 2026-10-23 with
   CPython 3.12.10 (Windows): ``-I`` implies ``-E``, so the interpreter ignores every PYTHON*
   variable, PYTHONHASHSEED included; under the registered invocation
   ``sys.flags.hash_randomization`` is 1 and ``hash("abc")`` differs between runs. The status
   records that effective state (``hash_randomization``, ``ignore_environment``) as a fact; it is
   not a pin, because the registered invocation cannot set it. v3 outputs must therefore never
   depend on the string hash seed; the determinism test runs the DEV recipe in fresh ``-I``
   interpreters and requires byte-identical bytes. Raised as an amendment request
   (``v3/AMENDMENT_REQUESTS.md``, AR-4); until it is decided this reading of the registered text
   applies. DEV runs record the status and label an unpinned interpreter; they are not
   registered computations.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

from . import registry


class GuardError(PermissionError):
    """Base class of every v3 guard refusal."""


class V2SealRefused(GuardError):
    """The v2 seal guard refused a namespace / split combination."""


class SealedPathRefused(GuardError):
    """A path under v2/data/sealed/ (or any 'sealed' directory) was refused."""


class DevRuleError(GuardError):
    """The DEV rule of s.15.5 was violated."""


class RuntimePinError(GuardError):
    """A registered computation does not run under the runtime pins of s.13 step 9."""


# --------------------------------------------------------------------------------------------
# name folding
# --------------------------------------------------------------------------------------------
_SEPARATORS = re.compile(r"[\s\-]+")


def fold_name(name: object) -> str:
    """NFKC, format characters (category Cf) removed, casefolded, separator runs to '_'."""
    text = unicodedata.normalize("NFKC", str(name))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return _SEPARATORS.sub("_", text.casefold().strip()).strip("_")


_V2_NAMESPACE = fold_name(registry.V2_SEED_NAMESPACE)
_V3_NAMESPACE = fold_name(registry.SEED_NAMESPACE)
_V2_SEALED = frozenset(fold_name(s) for s in registry.V2_SEALED_SPLITS)


def is_v2_sealed_name(split: object) -> bool:
    folded = fold_name(split)
    return folded in _V2_SEALED or folded == "test" or folded.startswith("shift")


def check_namespace(namespace: object, split: object) -> str:
    """Return the canonical namespace, or refuse (s.3.2 item 5)."""
    folded = fold_name(namespace)
    if folded == _V2_NAMESPACE:
        if is_v2_sealed_name(split):
            raise V2SealRefused(
                "namespace oac-ops-health-v2 with a v2 sealed split name is refused "
                "(v2 seal guard, prereg s.3.2 item 5)"
            )
        if namespace != registry.V2_SEED_NAMESPACE:
            raise V2SealRefused("the v2 namespace is accepted only in its exact registered spelling")
        return registry.V2_SEED_NAMESPACE
    if folded == _V3_NAMESPACE and namespace == registry.SEED_NAMESPACE:
        return registry.SEED_NAMESPACE
    raise V2SealRefused("unregistered seed namespace; only oac-ops-health-v3 (and v2 for parity) exist")


# --------------------------------------------------------------------------------------------
# sealed-path guard
# --------------------------------------------------------------------------------------------
def _norm(path: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _under(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory.rstrip(os.sep) + os.sep)


def is_sealed_path(path: Path | str) -> bool:
    raw = Path(path)
    absolute = Path(os.path.abspath(raw))
    try:
        resolved = raw.resolve(strict=False)
    except (OSError, RuntimeError):
        return True  # unresolvable (for example a symlink loop): refuse
    sealed_dir = registry.V2_SEALED_DIR
    targets = {_norm(sealed_dir), _norm(Path(os.path.abspath(sealed_dir)))}
    try:
        targets.add(_norm(sealed_dir.resolve(strict=False)))
    except (OSError, RuntimeError):
        pass
    for candidate in (absolute, resolved):
        normed = _norm(candidate)
        if any(_under(normed, t) for t in targets):
            return True
        if any(fold_name(part) == "sealed" for part in candidate.parts):
            return True
    return False


def refuse_sealed_path(path: Path | str) -> Path:
    if is_sealed_path(path):
        raise SealedPathRefused(
            "refusing a sealed path: v3 never reads v2/data/sealed/ or any 'sealed' directory "
            "(prereg s.3.2 item 5)"
        )
    return Path(path)


def read_bytes_guarded(path: Path | str) -> bytes:
    """Every v3 data read goes through here: the guard runs before the file is opened."""
    return refuse_sealed_path(path).read_bytes()


# --------------------------------------------------------------------------------------------
# DEV rule
# --------------------------------------------------------------------------------------------
def check_dev_split(split: object) -> None:
    if split != registry.DEV_SPLIT:
        raise DevRuleError("DEV rows exist only under the split name 'dev' (prereg s.15.5)")


class DevLedger:
    """Rows written per regime in this DEV run (one process)."""

    def __init__(self, cap: int = registry.DEV_MAX_ROWS_PER_REGIME):
        self.cap = cap
        self._streams: dict[tuple, tuple[str, int]] = {}
        self._lock = threading.Lock()

    def record(self, *, namespace: str, master_seed: int, regime: str, mode: str,
               draws: int, rows_per_draw: int) -> None:
        if draws < 0 or rows_per_draw not in (1, 2):
            raise DevRuleError("invalid DEV accounting")
        key = (namespace, master_seed, regime, mode)
        with self._lock:
            _bucket, previous = self._streams.get(key, (regime, 0))
            rows = max(previous, draws * rows_per_draw)
            total = self.total(regime) - previous + rows
            if total > self.cap:
                raise DevRuleError(
                    f"DEV cap exceeded: {total} rows of regime {regime!r} in this DEV run "
                    f"(at most {self.cap}, prereg s.15.5)"
                )
            self._streams[key] = (regime, rows)

    def total(self, regime: str) -> int:
        return sum(rows for bucket, rows in self._streams.values() if bucket == regime)

    def snapshot(self) -> dict:
        with self._lock:
            per_regime: dict[str, int] = {}
            for bucket, rows in self._streams.values():
                per_regime[bucket] = per_regime.get(bucket, 0) + rows
            streams = [
                {"namespace": k[0], "master_seed": k[1], "regime": k[2], "mode": k[3], "rows": v[1]}
                for k, v in sorted(self._streams.items())
            ]
        return {"cap_per_regime": self.cap, "rows_per_regime": dict(sorted(per_regime.items())),
                "streams": streams}


DEV_LEDGER = DevLedger()


def is_dev_row(row: Mapping) -> bool:
    sample_id = row.get("sample_id") if isinstance(row, Mapping) else None
    return isinstance(sample_id, str) and sample_id.startswith(registry.DEV_SPLIT + "-")


def check_dev_fit(rows: Iterable[Mapping]) -> int:
    """Refuse a fit on more than 2,048 DEV rows, or on DEV rows mixed with other rows."""
    rows = list(rows)
    dev = sum(1 for row in rows if is_dev_row(row))
    if dev and dev != len(rows):
        raise DevRuleError("a fit may not mix DEV rows with rows of a registered split")
    if dev > registry.DEV_MAX_FIT_ROWS:
        raise DevRuleError(
            f"no fit uses more than {registry.DEV_MAX_FIT_ROWS} rows of DEV data (prereg s.15.5)"
        )
    return dev


def refuse_dev_statistic(split_names: Iterable[str], what: str) -> None:
    """No P or H statistic, tolerance, detection ratio, power or WIN-chance quantity on DEV data."""
    for name in split_names:
        if not isinstance(name, str) or fold_name(name).startswith(registry.DEV_SPLIT):
            raise DevRuleError(f"{what} is never computed on DEV data (prereg s.15.5)")


# --------------------------------------------------------------------------------------------
# runtime pins
# --------------------------------------------------------------------------------------------
def runtime_pin_status() -> dict:
    version = tuple(sys.version_info[:3])
    hashseed = os.environ.get("PYTHONHASHSEED")
    status = {
        "implementation": sys.implementation.name,
        "version": ".".join(map(str, version)),
        "version_pinned": sys.implementation.name == "cpython" and version == registry.PINNED_PYTHON,
        "isolated": bool(sys.flags.isolated),
        "pythonhashseed": hashseed,
        "pythonhashseed_pinned": hashseed == registry.PINNED_HASHSEED,
        # effective state, recorded as a fact (not a pin): under -I the variable is ignored
        "ignore_environment": bool(sys.flags.ignore_environment),
        "hash_randomization": int(sys.flags.hash_randomization),
    }
    status["pinned"] = (
        status["version_pinned"] and status["isolated"] and status["pythonhashseed_pinned"]
    )
    return status


def require_runtime_pins() -> dict:
    status = runtime_pin_status()
    if not status["pinned"]:
        raise RuntimePinError(
            "registered computations run only as CPython 3.12.10 'python -I' with PYTHONHASHSEED=0 "
            f"(prereg s.13 step 9); found {status}"
        )
    return status
