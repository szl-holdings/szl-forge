"""Paired percentile bootstrap (PREREGISTRATION section 7).  Standard library only.

Resampling
  One ``random.Random(seed)`` drives all resamples.  For resample b = 1..B and each stratum in
  the given order (default: one stratum = all rows), m row indices are drawn with replacement
  from the stratum's m rows as int(random() * m) (the same draw ``random.choices`` makes),
  evaluated with C-level builtins (itertools.starmap / map).  The draws are turned into a
  per-resample count-weight vector c (c_i = times row i was drawn, sum c_i = n) with
  collections.Counter and dict.get.  Every contender and every statistic is evaluated on the
  SAME count vector (paired); only one vector is alive at a time.  With strata (for example the
  four shift regimes of the pooled suite), each stratum keeps its size in every resample.
Weighted metrics
  A statistic of a resample equals the statistic of the multiset of drawn rows, computed from
  count-weighted sums: integer sums via sum(itertools.compress(...)) and float sums via
  math.fsum(map(operator.mul, ...)).  With c = all ones they equal metrics.py exactly (tests).
  Weighted AUROC uses score-sorted prefix sums of negative weight (itertools.accumulate):
  AUC = sum_{i pos} c_i (N_below(i) + N_upto(i)) / (2 P N), which is the Mann-Whitney statistic
  with ties counted 1/2, in exact integer arithmetic.
Intervals
  Percentile interval at level 0.95: the 2.5th and 97.5th percentiles of the B resample values,
  with linear interpolation between order statistics (Hyndman-Fan type 7: h = (B - 1) p).
  Resamples where a statistic is undefined (None) are excluded and counted in n_undefined.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import accumulate, compress, repeat, starmap
from operator import add, itemgetter, mul
from typing import Callable, Iterator, Mapping, Sequence

from . import metrics

RESAMPLES = 2000
LEVEL = 0.95


def _pick(seq: Sequence, indices: Sequence[int]) -> tuple:
    """itemgetter(*indices)(seq) that always returns a tuple."""
    if not indices:
        return ()
    if len(indices) == 1:
        return (seq[indices[0]],)
    return itemgetter(*indices)(seq)


class Resampler:
    """Deterministic stream of count-weight vectors (paired across contenders)."""

    def __init__(
        self,
        n_rows: int,
        *,
        seed: int,
        resamples: int = RESAMPLES,
        strata: Sequence[Sequence[int]] | None = None,
    ):
        if n_rows <= 0:
            raise ValueError("n_rows must be positive")
        self.n = n_rows
        self.seed = seed
        self.resamples = resamples
        if strata is None:
            self.strata = None
        else:
            flat = sorted(i for stratum in strata for i in stratum)
            if flat != list(range(n_rows)):
                raise ValueError("strata must partition 0..n_rows-1")
            if any(len(s) == 0 for s in strata):
                raise ValueError("strata must be non-empty")
            self.strata = [tuple(s) for s in strata]

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        rnd = rng.random
        n = self.n
        index_range = range(n)
        for _ in range(self.resamples):
            if self.strata is None:
                draws = map(int, map(mul, repeat(float(n), n), starmap(rnd, repeat((), n))))
                counts = Counter(draws)
            else:
                counts = Counter()
                for stratum in self.strata:
                    m = len(stratum)
                    local = map(int, map(mul, repeat(float(m), m), starmap(rnd, repeat((), m))))
                    counts.update(map(stratum.__getitem__, local))
            yield list(map(counts.get, index_range, repeat(0)))


class BinaryStats:
    """Count-weighted confusion-derived metrics for one contender's 0/1 decisions."""

    def __init__(self, labels: Sequence[int], decisions: Sequence[int]):
        y = [int(v) for v in labels]
        d = [int(v) for v in decisions]
        if len(y) != len(d) or any(v not in (0, 1) for v in y + d):
            raise ValueError("labels/decisions must be aligned 0/1 sequences")
        self.pos = bytes(y)
        self.tp = bytes(a & b for a, b in zip(y, d))
        self.tn = bytes((1 - a) & (1 - b) for a, b in zip(y, d))

    def counts(self, w: Sequence[int]) -> dict[str, int]:
        total = sum(w)
        p = sum(compress(w, self.pos))
        tp = sum(compress(w, self.tp))
        tn = sum(compress(w, self.tn))
        return {"tp": tp, "fp": (total - p) - tn, "tn": tn, "fn": p - tp}

    def rates(self, w: Sequence[int]) -> dict[str, float | None]:
        c = self.counts(w)
        return metrics.rates_from_counts(c["tp"], c["fp"], c["tn"], c["fn"])

    def balanced_accuracy(self, w: Sequence[int]) -> float | None:
        return self.rates(w)["balanced_accuracy"]


class ScoreStats:
    """Count-weighted Brier, log loss, ECE and AUROC for one contender's scores."""

    def __init__(self, labels: Sequence[int], scores: Sequence[float], bins: int = metrics.ECE_BINS):
        y = [int(v) for v in labels]
        s = [float(v) for v in scores]
        if len(y) != len(s):
            raise ValueError("labels and scores must be aligned")
        n = len(y)
        self.n = n
        self.sq = [(si - yi) ** 2 for si, yi in zip(s, y)]
        self.ll = metrics.log_loss_terms(y, s)
        # ECE: rows permuted so that bins are contiguous; d_i = s_i - y_i.
        bin_of = [metrics.ece_bin(si, bins) for si in s]
        self.ece_perm = sorted(range(n), key=lambda i: (bin_of[i], i))
        self.ece_diff = [s[i] - y[i] for i in self.ece_perm]
        self.ece_slices = []
        start = 0
        for b in range(bins):
            end = start
            while end < n and bin_of[self.ece_perm[end]] == b:
                end += 1
            self.ece_slices.append((start, end))
            start = end
        # AUROC: score order, tie groups, positive rows.
        order = sorted(range(n), key=lambda i: (s[i], i))
        self.auc_order = order
        self.auc_neg_sorted = bytes(1 - y[i] for i in order)
        group_start = [0] * n
        group_end = [0] * n
        k = 0
        while k < n:
            j = k
            while j + 1 < n and s[order[j + 1]] == s[order[k]]:
                j += 1
            for m in range(k, j + 1):
                group_start[order[m]] = k
                group_end[order[m]] = j + 1
            k = j + 1
        self.pos_rows = [i for i in range(n) if y[i] == 1]
        self.pos_lo = [group_start[i] for i in self.pos_rows]
        self.pos_hi = [group_end[i] for i in self.pos_rows]
        self.pos_mask = bytes(y)

    def brier(self, w: Sequence[int]) -> float | None:
        total = sum(w)
        return math.fsum(map(mul, w, self.sq)) / total if total else None

    def log_loss(self, w: Sequence[int]) -> float | None:
        total = sum(w)
        return math.fsum(map(mul, w, self.ll)) / total if total else None

    def ece(self, w: Sequence[int]) -> float | None:
        total = sum(w)
        if not total:
            return None
        wp = _pick(w, self.ece_perm)
        gaps = [
            abs(math.fsum(map(mul, wp[a:b], self.ece_diff[a:b]))) for a, b in self.ece_slices
        ]
        return math.fsum(gaps) / total

    def auroc(self, w: Sequence[int]) -> float | None:
        p = sum(compress(w, self.pos_mask))
        neg = sum(w) - p
        if p == 0 or neg == 0:
            return None
        neg_sorted = map(mul, _pick(w, self.auc_order), self.auc_neg_sorted)
        cum = list(accumulate(neg_sorted, initial=0))
        wpos = _pick(w, self.pos_rows)
        lo = _pick(cum, self.pos_lo)
        hi = _pick(cum, self.pos_hi)
        num2 = sum(map(mul, wpos, map(add, lo, hi)))
        return num2 / (2 * p * neg)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Type-7 percentile of an ascending sequence."""
    if not sorted_values:
        raise ValueError("no values")
    h = (len(sorted_values) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def interval(values: Sequence[float | None], level: float = LEVEL) -> dict:
    defined = sorted(v for v in values if v is not None)
    tail = (1.0 - level) / 2.0
    return {
        "level": level,
        "lower": percentile(defined, tail) if defined else None,
        "upper": percentile(defined, 1.0 - tail) if defined else None,
        "n_defined": len(defined),
        "n_undefined": len(values) - len(defined),
        "method": "percentile_type7",
    }


def paired_bootstrap(
    n_rows: int,
    statistics: Mapping[str, Callable[[Sequence[int]], float | None]],
    *,
    seed: int,
    resamples: int = RESAMPLES,
    strata: Sequence[Sequence[int]] | None = None,
) -> dict[str, list[float | None]]:
    """Evaluate every statistic on the same count vectors; returns per-statistic value lists."""
    names = list(statistics)
    out: dict[str, list[float | None]] = {name: [] for name in names}
    for w in Resampler(n_rows, seed=seed, resamples=resamples, strata=strata):
        for name in names:
            out[name].append(statistics[name](w))
    return out


def differences(a: Sequence[float | None], b: Sequence[float | None]) -> list[float | None]:
    """Paired per-resample differences a - b (None if either is undefined)."""
    return [None if x is None or y is None else x - y for x, y in zip(a, b, strict=True)]


def summarize(point: float | None, values: Sequence[float | None], level: float = LEVEL) -> dict:
    return {"point": point, **interval(values, level)}
