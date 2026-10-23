"""The registered bootstrap of protocol s.8, reusing v2's paired percentile bootstrap. SYNTHETIC.

v2's implementation (``v2/research/bootstrap.py``) is reused unchanged: the count-vector
``Resampler`` (draws ``int(random() * m)`` per stratum, in stratum order, one
``random.Random(seed)`` per stream), the count-weighted ``BinaryStats`` and ``ScoreStats``, the
type-7 percentile interval, ``paired_bootstrap``, ``differences`` and ``summarize``. v3 registers
the layout (s.8):

  * 2,000 paired resamples; percentile intervals, type 7, at 0.95;
  * one stream per evaluation set, each a fresh ``random.Random(20261014)``;
  * the evaluation sets and their strata, in this order:
      1. ``test``: one stratum of 100,000 rows;
      2. the pooled shift suite: four strata of 25,000 rows each, in the order
         shift_queue_saturation, shift_failure_bursts, shift_clock_skew, shift_missing_tls
         (the per-regime secondary intervals reuse each stratum's counts);
      3. the H3 twin set: one stratum of 100,000 PAIR indices; each drawn index gives its count
         to both rows of the pair (``pair_counts``);
      4. ``legacy_v1_test``: one stratum of 240 rows;
  * a resample on which a statistic is undefined is excluded and counted (v2's ``interval``);
  * the FRAGILE check's 100 further stream sets: set i uses ``random.Random(20261014 + i)``,
    i = 1..100, in the same per-set layout (``fragile_seeds``).

Sizes are parameters so that the layout can be tested on DEV-sized inputs; the registered sizes
are the defaults. No P or H statistic is computed on DEV data (s.15.5).
"""

from __future__ import annotations

from typing import Iterator, Sequence

from v2.research import bootstrap as v2_bootstrap

from . import registry

SEED = 20261014
RESAMPLES = 2000
LEVEL = 0.95
FRAGILE_SETS = 100
TEST_ROWS = 100000
SHIFT_ROWS = 25000
H3_PAIRS = 100000
LEGACY_ROWS = 240
EVALUATION_SETS = ("test", "pooled_shift", "h3_twins", "legacy_v1_test")

Resampler = v2_bootstrap.Resampler
BinaryStats = v2_bootstrap.BinaryStats
ScoreStats = v2_bootstrap.ScoreStats
percentile = v2_bootstrap.percentile
interval = v2_bootstrap.interval
paired_bootstrap = v2_bootstrap.paired_bootstrap
differences = v2_bootstrap.differences
summarize = v2_bootstrap.summarize


def pooled_strata(sizes: Sequence[int] = (SHIFT_ROWS,) * 4) -> list[range]:
    """Consecutive index ranges, one per shift split, in POOLED_SHIFT_SUITE order."""
    if len(sizes) != len(registry.POOLED_SHIFT_SUITE):
        raise ValueError("the pooled suite has exactly four strata")
    strata, start = [], 0
    for size in sizes:
        strata.append(range(start, start + size))
        start += size
    return strata


def resampler(evaluation_set: str, *, sizes: Sequence[int] | None = None, seed: int = SEED,
              resamples: int = RESAMPLES) -> Resampler:
    """A FRESH stream for one evaluation set (s.8: one stream per set)."""
    if evaluation_set == "test":
        n = sizes[0] if sizes else TEST_ROWS
        return Resampler(n, seed=seed, resamples=resamples)
    if evaluation_set == "pooled_shift":
        strata = pooled_strata(sizes if sizes else (SHIFT_ROWS,) * 4)
        n = sum(len(s) for s in strata)
        return Resampler(n, seed=seed, resamples=resamples, strata=[list(s) for s in strata])
    if evaluation_set == "h3_twins":
        n = sizes[0] if sizes else H3_PAIRS
        return Resampler(n, seed=seed, resamples=resamples)
    if evaluation_set == "legacy_v1_test":
        n = sizes[0] if sizes else LEGACY_ROWS
        return Resampler(n, seed=seed, resamples=resamples)
    raise ValueError(f"unknown evaluation set {evaluation_set!r}")


def pair_counts(pair_vector: Sequence[int]) -> tuple[list[int], list[int]]:
    """Counts of the observed and the latent twin file: each drawn pair index counts for both."""
    counts = list(pair_vector)
    return counts, list(counts)


def regime_counts(pooled_vector: Sequence[int], sizes: Sequence[int] = (SHIFT_ROWS,) * 4) -> dict:
    """Per-regime counts from the pooled resample (secondary per-regime intervals reuse them)."""
    return {name: list(pooled_vector[s.start:s.stop])
            for name, s in zip(registry.POOLED_SHIFT_SUITE, pooled_strata(sizes), strict=True)}


def fragile_seeds() -> list[int]:
    return [SEED + i for i in range(1, FRAGILE_SETS + 1)]


def stream(evaluation_set: str, **kwargs) -> Iterator[list[int]]:
    return iter(resampler(evaluation_set, **kwargs))
