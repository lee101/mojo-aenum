"""Array-shaped versions of aenum's flag arithmetic.

aenum evaluates the operations in this module one flag member at a time, on
Python ints. The kernels evaluate the identical integer expressions over a
whole array of flag values, which is the only way they are worth compiling.
Every function here is exact: these are integer and bitwise operations, so
results are bit-for-bit identical to aenum's.
"""

import numpy as np

from . import _lib

__all__ = [
    "split",
    "combine",
    "contains",
    "flag_len",
    "is_single_bit",
    "high_bit",
    "iter_flag_values",
    "next_flag_value",
]


def _values(values):
    return np.ascontiguousarray(values, dtype=np.int64)


def split(values, mask=None):
    """The set bits of every value, lowest first: `aenum.split` per value."""
    bits, counts = _lib.split(values, mask=mask)
    return [bits[i, :counts[i]].tolist() for i in range(bits.shape[0])]


def combine(values, mask=0, op="or"):
    """`v | mask`, `v & mask`, `v ^ mask`, `~v` or `~v & mask`, per value."""
    return _lib.combine(values, mask, op)


def contains(values, other):
    """`Flag.__contains__`: does each value have at least `other`'s bits?

    Zero is never contained in anything and contains nothing, matching
    aenum's short circuit.
    """
    return _lib.contains(values, other)


def flag_len(values):
    """`len(flag)` per value: the number of bits set."""
    return _lib.popcount(values)


def is_single_bit(values):
    """`aenum.is_single_bit` per value."""
    return _lib.single_bit(values)


def high_bit(values):
    """`aenum._enum._high_bit` per value: -1 for zero."""
    return _lib.high_bit(values)


def iter_flag_values(values, singles_mask):
    """The member values a flag iteration would visit, in upstream's order.

    `Flag.__iter__` walks `_iter_bits_lsb(value & cls._singles_mask_)` and
    looks each bit up in the class's value-to-member map, so the numeric core
    is the masked LSB-first split.
    """
    return split(values, mask=singles_mask)


def next_flag_value(last_values, count):
    """`Flag._generate_next_value_`: the next power of two above the last value.

    `count` is the number of members already defined; upstream returns 1 for
    the first member and `2 ** (high_bit(max(last_values)) + 1)` afterwards.
    """
    return _lib.next_flag_value(last_values, count)


def _as_int_list(values):
    """Accept Python ints, numpy scalars or an array-like of flag values."""
    arr = _values(values)
    if arr.ndim != 1:
        raise ValueError("flag values must be a one-dimensional sequence")
    return arr
