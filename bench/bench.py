"""Correctness-gated benchmark for mojo-aenum.

Every case checks the Mojo result against the real aenum (or against the exact
integer expression it is defined as) before timing, so a wrong kernel shows up
as a correctness failure rather than as a suspiciously good number.

Baselines are the fastest reasonable vectorised forms, not Python loops:
NumPy elementwise bitwise for the combine operations, a SWAR popcount for the
population count, `np.frexp` for the highest-set-bit index. The split case is
the exception: there is no vectorised way to yield set bits one at a time, so
the baseline is aenum's own generator, and that is labelled in the output.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_aenum  # noqa: E402
from aenum._enum import _iter_bits_lsb  # noqa: E402

N = 1 << 22


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _values(seed=0, n=N):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << 40, size=n, dtype=np.int64)


def s(n):
    return f"n={n}"


_NUMPY_OPS = {
    "or": lambda v, m: v | m,
    "and": lambda v, m: v & m,
    "xor": lambda v, m: v ^ m,
    "invert": lambda v, m: ~v,
}


def bench_combine(op, n=N):
    v = _values(1, n)
    m = int(0x5A5A5A5A)
    reference = _NUMPY_OPS[op]
    got = mojo_aenum.combine(v, m, op)
    np.testing.assert_array_equal(got, reference(v, m))
    numpy_time = _time(lambda: reference(v, m))
    mojo_time = _time(lambda: mojo_aenum.combine(v, m, op))
    return f"combine {op} {s(n)}", numpy_time, mojo_time


def bench_contains(n=N):
    v = _values(2, n)
    other = int(0x0F0F0F0F)
    got = mojo_aenum.contains(v, other)
    expect = (v & other) == other
    np.testing.assert_array_equal(got, expect)
    numpy_time = _time(lambda: (v & other) == other)
    mojo_time = _time(lambda: mojo_aenum.contains(v, other))
    return f"contains {s(n)}", numpy_time, mojo_time


def _swar_popcount(v):
    x = v.copy()
    x = x - ((x >> np.int64(1)) & np.int64(0x5555555555555555))
    x = (x & np.int64(0x3333333333333333)) + (
        (x >> np.int64(2)) & np.int64(0x3333333333333333)
    )
    x = (x + (x >> np.int64(4))) & np.int64(0x0F0F0F0F0F0F0F0F)
    return (x * np.int64(0x0101010101010101)) >> np.int64(56)


def bench_flag_len(n=N):
    v = _values(3, n)
    got = mojo_aenum.flag_len(v)
    expect = _swar_popcount(v)
    np.testing.assert_array_equal(got, expect)
    numpy_time = _time(lambda: _swar_popcount(v))
    mojo_time = _time(lambda: mojo_aenum.flag_len(v))
    return f"flag_len (popcount) {s(n)}", numpy_time, mojo_time


def bench_high_bit(n=N):
    v = _values(4, n)
    got = mojo_aenum.high_bit(v)
    frexp = np.frexp(v.astype(np.float64))[1]
    expect = np.where(v > 0, frexp - 1, -1).astype(np.int64)
    np.testing.assert_array_equal(got, expect)
    numpy_time = _time(lambda: np.where(v > 0, frexp - 1, -1))
    mojo_time = _time(lambda: mojo_aenum.high_bit(v))
    return f"high_bit {s(n)}", numpy_time, mojo_time


def bench_split(n=1 << 18):
    # fewer elements: the per-value output is a variable-length list, so the
    # comparison is dominated by list building, not by the kernel
    v = _values(5, n) & np.int64(0xFF)
    got = mojo_aenum.split(v)
    theirs = [list(_iter_bits_lsb(int(x))) for x in v]
    assert got == theirs
    numpy_time = _time(lambda: [list(_iter_bits_lsb(int(x))) for x in v], 3)
    mojo_time = _time(lambda: mojo_aenum.split(v), 3)
    return f"split (vs aenum generator) {s(n)}", numpy_time, mojo_time


def main():
    print(f"{'case':<42}{'baseline':>14}{'mojo-aenum':>14}{'ratio':>9}")
    print("-" * 79)
    cases = [
        lambda: bench_combine("or"),
        lambda: bench_combine("and"),
        lambda: bench_combine("xor"),
        lambda: bench_combine("invert"),
        lambda: bench_contains(),
        lambda: bench_flag_len(),
        lambda: bench_high_bit(),
        lambda: bench_split(),
    ]
    for case in cases:
        label, ref, got = case()
        ratio = ref / got if got else float("nan")
        print(f"{label:<42}{ref*1e3:>12.2f}ms{got*1e3:>12.2f}ms{ratio:>8.2f}x")


if __name__ == "__main__":
    main()
