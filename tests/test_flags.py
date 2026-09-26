"""Parity tests for the Mojo flag-algebra kernels against the real aenum.

aenum applies every operation here to a single flag member, aenum applies them
to Python ints, so the reference is aenum itself: `aenum.split`,
`aenum.is_single_bit`, `aenum._enum._high_bit`, and the operators, `__len__`,
`__contains__` and `__iter__` of a real `Flag` subclass. Bitwise and integer
arithmetic is exact, so every assertion here is `==` with no tolerance.
"""

import numpy as np
import pytest

import aenum
import mojo_aenum
from aenum import Flag, auto
from aenum._enum import _high_bit as ref_high_bit
from aenum._enum import bit_len as ref_bit_len
from aenum._enum import is_single_bit as ref_single_bit
from aenum._enum import show_flag_values as ref_split

# aenum's Python ints are unbounded; the kernels work in int64, so parity is
# asserted over everything a real Flag class can hold (powers of two up to
# 2**62) and the 2**63 boundary is covered by its own test below.
VALUES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 127, 128,
    255, 256, 1023, 4095, 65535, 1 << 20, (1 << 31) - 1, 1 << 31, 1 << 32,
    1 << 61, 1 << 62, (1 << 62) - 1, (1 << 62) | 5, (1 << 40) + 12345,
]


# IntFlag is the KEEP-boundary Flag: a strict Flag refuses values outside its
# declared mask, which would make most of the parity values illegal. Its
# __contains__/__len__/__iter__/operators are Flag's, with int operands
# additionally allowed.
class Perm(aenum.IntFlag):
    READ = 1
    WRITE = 2
    EXECUTE = 4
    OWNER = 8


@pytest.mark.parametrize("value", VALUES)
def test_split_matches_aenum(value):
    assert mojo_aenum.split([value]) == [ref_split(value)]


def test_split_over_a_batch_matches_aenum():
    assert mojo_aenum.split(VALUES) == [ref_split(v) for v in VALUES]


def test_split_is_lowest_bit_first():
    # aenum yields set bits from the least significant upwards; an
    # implementation that sorted descending, or that returned the bit indices
    # instead of the bit values, would fail here
    assert mojo_aenum.split([0b1011]) == [[1, 2, 8]]
    assert ref_split(0b1011) == [1, 2, 8]


def test_split_with_mask_matches_masked_iter_bits():
    # Flag.__iter__ restricts to single-bit members via _singles_mask_
    mask = Perm.READ.value | Perm.WRITE.value
    got = mojo_aenum.split([0b1101], mask=mask)
    theirs = ref_split(0b1101 & mask)
    assert got == [theirs]


def test_split_rejects_negative_values():
    # upstream's `value ^= bit` leaves a negative value negative, so
    # _iter_bits_lsb never terminates; the shim refuses instead of hanging
    with pytest.raises(ValueError):
        mojo_aenum.split([-4])


def test_split_on_empty_batch():
    assert mojo_aenum.split([]) == []


def test_combine_or_and_xor_match_flag_operators():
    mask = 0b1011
    theirs = {
        "or": [(Perm(v) | Perm(mask)).value for v in VALUES],
        "and": [(Perm(v) & Perm(mask)).value for v in VALUES],
        "xor": [(Perm(v) ^ Perm(mask)).value for v in VALUES],
    }
    for op, expect in theirs.items():
        assert mojo_aenum.combine(VALUES, mask, op).tolist() == expect


def test_combine_invert_masked_matches_flag_invert():
    # aenum inverts inside the class's single-bit mask, so `~Perm.READ` is
    # 0b1110 rather than a 64-bit complement; the masked form is the
    # faithful one. Only values inside the mask are comparable: for a value
    # with bits outside it, aenum's KEEP boundary complements over
    # max(bit_length(v), mask bits), which is a class-construction policy
    # rather than bit arithmetic.
    mask = (
        Perm.READ.value | Perm.WRITE.value | Perm.EXECUTE.value | Perm.OWNER.value
    )
    assert mask == 0b1111
    inside = [v for v in VALUES if v & ~mask == 0]
    got = mojo_aenum.combine(inside, mask, "invert_masked")
    assert got.tolist() == [(~v) & mask for v in inside]
    assert got.tolist() == [(~Perm(v)).value for v in inside]
    assert (~Perm.READ).value == 0b1110


def test_combine_invert_is_the_raw_complement():
    got = mojo_aenum.combine(VALUES, op="invert")
    assert got.tolist() == [~v for v in VALUES]


def test_combine_invert_masked_is_width_limited():
    mask = 0b1111
    got = mojo_aenum.combine([0b1010], mask, "invert_masked")
    assert got.tolist() == [(~0b1010) & mask]


def test_combine_rejects_unknown_op():
    with pytest.raises(ValueError):
        mojo_aenum.combine([1], 1, "nand")


@pytest.mark.parametrize("value", VALUES)
def test_popcount_matches_len_of_flag(value):
    got = mojo_aenum.flag_len([value])
    if value < (1 << 31):
        assert got.tolist() == [len(Perm(value))]
    # independent reference for the rest: popcount
    assert got.tolist() == [bin(value).count("1")]


def test_popcount_of_zero_is_zero():
    assert mojo_aenum.flag_len([0]).tolist() == [0]


@pytest.mark.parametrize("value", VALUES)
def test_single_bit_matches_aenum(value):
    assert mojo_aenum.is_single_bit([value]).tolist() == [
        ref_single_bit(value)
    ]


def test_single_bit_edges():
    assert mojo_aenum.is_single_bit([0, 1, 2, 3, 6, 1 << 62]).tolist() == [
        False, True, True, False, False, True
    ]


@pytest.mark.parametrize("value", VALUES)
def test_high_bit_matches_aenum(value):
    assert mojo_aenum.high_bit([value]).tolist() == [ref_high_bit(value)]


def test_high_bit_of_zero_is_minus_one():
    assert mojo_aenum.high_bit([0]).tolist() == [-1]
    assert ref_high_bit(0) == -1


@pytest.mark.parametrize("value", VALUES)
def test_high_bit_plus_one_is_bit_len(value):
    # aenum's bit_len is a shift loop; a kernel that counted one too few or
    # too many iterations would disagree with it
    if value == 0:
        return
    assert mojo_aenum.high_bit([value]).tolist()[0] + 1 == ref_bit_len(value)


@pytest.mark.parametrize("other", [1, 2, 3, 4, 8, 9, 12])
def test_contains_matches_flag_contains(other):
    got = mojo_aenum.contains(VALUES, other)
    theirs = [other in Perm(v) for v in VALUES if v < (1 << 31)]
    assert got[: len(theirs)].tolist() == theirs


def test_contains_zero_is_false_like_aenum():
    # aenum short-circuits: a zero flag is not contained in anything, and
    # nothing is contained in a zero flag
    class Bits(aenum.IntFlag):
        A = 1
        B = 2

    got = mojo_aenum.contains([Bits.A.value, 0, Bits.A.value | Bits.B.value], 0)
    assert got.tolist() == [False, False, False]
    assert (0 in Bits.A) is False
    assert (Bits.A in Bits(0)) is False
    assert mojo_aenum.contains([Bits.A.value | Bits.B.value], 0).tolist() == [False]


def test_iter_flag_values_matches_flag_iteration():
    combined = [
        (Perm.READ | Perm.EXECUTE).value,
        (Perm.READ | Perm.WRITE | Perm.OWNER).value,
        (Perm.WRITE | Perm.OWNER).value,
        0,
    ]
    got = mojo_aenum.iter_flag_values(combined, 0b1111)
    theirs = [[m.value for m in Perm(v)] for v in combined]
    assert got == theirs


def test_next_flag_value_matches_auto_generated_members():
    # aenum's Flag._generate_next_value_ hands out 1, 2, 4, 8, ... and a
    # kernel that used the previous value instead of the next power of two
    # would repeat a bit
    class Auto(Flag):
        A = auto()
        B = auto()
        C = auto()
        D = auto()

    values = [m.value for m in Auto]
    assert values == [1, 2, 4, 8]
    got = mojo_aenum.next_flag_value([0, 1, 2, 4], 0)
    assert got.tolist() == [1, 1, 1, 1]
    got = mojo_aenum.next_flag_value([1, 2, 4, 8], 1)
    assert got.tolist() == [2, 4, 8, 16]


def test_next_flag_value_after_a_combined_member():
    # a class whose last member is a combination must still take a fresh bit
    assert mojo_aenum.next_flag_value([1 | 8], 1).tolist() == [16]
    assert 2 ** (ref_high_bit(1 | 8) + 1) == 16


def test_next_flag_value_top_bit():
    assert mojo_aenum.next_flag_value([1 << 60], 1).tolist() == [1 << 61]
    assert mojo_aenum.next_flag_value([(1 << 61) | 3], 1).tolist() == [1 << 62]


@pytest.mark.parametrize("last", [1 << 62, 1 << 63, (1 << 63) | 1])
def test_next_flag_value_beyond_64_bits_is_refused(last):
    # aenum would hand out 2**63 or 2**64 here; those are not signed int64, so
    # the shim says so rather than wrapping to a negative power of two
    with pytest.raises(OverflowError):
        mojo_aenum.next_flag_value([last], 1)


def test_values_wider_than_64_bits_are_refused():
    with pytest.raises(OverflowError):
        mojo_aenum.high_bit([1 << 64])


def test_empty_batches():
    assert mojo_aenum.combine([], 1, "or").size == 0
    assert mojo_aenum.flag_len([]).size == 0
    assert mojo_aenum.contains([], 1).size == 0
    assert mojo_aenum.high_bit([]).size == 0
    assert mojo_aenum.is_single_bit([]).size == 0
    assert mojo_aenum.next_flag_value([], 0).size == 0


def test_numpy_input_is_accepted():
    arr = np.array([0b1100, 0b0011], dtype=np.int64)
    assert mojo_aenum.combine(arr, 0b1000, "or").tolist() == [12, 11]
    assert mojo_aenum.split(arr) == [ref_split(12), ref_split(3)]


def test_input_array_is_not_mutated():
    arr = np.array([0b1100, 0b0011], dtype=np.int64)
    before = arr.copy()
    mojo_aenum.combine(arr, 0b1000, "xor")
    mojo_aenum.split(arr)
    np.testing.assert_array_equal(arr, before)
