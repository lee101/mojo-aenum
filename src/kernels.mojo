"""Integer bit algebra: the only arithmetic aenum's `Flag` actually performs.

aenum is a type-system library. Its 3,382 lines are enum machinery: member
collection, `_generate_next_value_` dispatch, named tuples, descriptors,
aliases. The numeric surface is the flag algebra and nothing else, and it is
exactly this file's worth of integer operations:

- `Flag.__or__/__and__/__xor__`  -> `value | other`, `value & other`, `value ^ other`
- `Flag.__invert__`              -> `~value`
- `Flag.__contains__`            -> `other & value == other`
- `Flag.__len__`                 -> `bit_count(value)`
- `_iter_bits_lsb` / `split`     -> `bit = value & (~value + 1); value ^= bit`
- `is_single_bit`                -> `num != 0 and num & (num - 1) == 0`
- `_high_bit`                    -> `value.bit_length() - 1`
- `Flag._generate_next_value_`   -> `2 ** (_high_bit(max(values)) + 1)`

aenum evaluates all of these one scalar at a time, on a single member. These
kernels evaluate the identical integer expressions over a whole array of flag
values, which is the only honest way to make them worth a shared library.

Everything here is exact: integer and bitwise operations have no FMA, so the
parity tests assert exact equality.

Buffers cross the C ABI as 64-bit addresses, `@export` rejects parametric
functions, and `AnyOrigin[mut=True]` is the only usable mutable origin, so each
export takes `Int` addresses and rebuilds its own pointer.
"""

comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

# combine operations, shared with the Python shim
comptime OP_OR: Int = 0
comptime OP_AND: Int = 1
comptime OP_XOR: Int = 2
comptime OP_INVERT: Int = 3
comptime OP_INVERT_MASKED: Int = 4


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def highest_bit(v: Int64) -> Int:
    """Index of the highest set bit, or -1 for a non-positive value.

    aenum's `_high_bit` is `value.bit_length() - 1` and documents that it
    returns -1 for zero or negative input; this is that index, found with the
    standard six-step binary search instead of one shift per bit.
    """
    if v <= 0:
        return -1
    var n = 0
    var x = v
    if x > 0xFFFFFFFF:
        n += 32
        x = x >> Int64(32)
    if x > 0xFFFF:
        n += 16
        x = x >> Int64(16)
    if x > 0xFF:
        n += 8
        x = x >> Int64(8)
    if x > 0xF:
        n += 4
        x = x >> Int64(4)
    if x > 0x3:
        n += 2
        x = x >> Int64(2)
    if x > 0x1:
        n += 1
    return n


@export("ae_flag_combine")
def ae_flag_combine(vals_addr: Int, mask: Int, n: Int, res_addr: Int,
                    op: Int) abi("C"):
    """`or`, `and`, `xor`, `~`, and `~value & mask` over n flag values.

    `INVERT` ignores `mask` and produces the full 64-bit complement, which is
    what `Flag.__invert__` computes. `INVERT_MASKED` is the width-limited form
    aenum uses when it has to render a complement in a fixed number of bits.
    """
    var vals = ip(vals_addr)
    var res = ip(res_addr)
    var m = Int64(mask)
    for i in range(n):
        var v = vals[unsafe_offset=i]
        if op == OP_OR:
            res[unsafe_offset=i] = v | m
        elif op == OP_AND:
            res[unsafe_offset=i] = v & m
        elif op == OP_XOR:
            res[unsafe_offset=i] = v ^ m
        elif op == OP_INVERT:
            res[unsafe_offset=i] = ~v
        else:
            res[unsafe_offset=i] = ~v & m


@export("ae_flag_popcount")
def ae_flag_popcount(vals_addr: Int, n: Int, res_addr: Int) abi("C"):
    """`Flag.__len__`: the number of bits set in each value."""
    var vals = ip(vals_addr)
    var res = ip(res_addr)
    for i in range(n):
        var v = vals[unsafe_offset=i]
        var count = 0
        while v != 0:
            v = v & (v - 1)
            count += 1
        res[unsafe_offset=i] = Int64(count)


@export("ae_flag_single_bit")
def ae_flag_single_bit(vals_addr: Int, n: Int, res_addr: Int) abi("C"):
    """`aenum.is_single_bit`: non-zero and with no adjacent set bit."""
    var vals = ip(vals_addr)
    var res = ip(res_addr)
    for i in range(n):
        var v = vals[unsafe_offset=i]
        var ok = Int64(0)
        if v != 0 and (v & (v - 1)) == 0:
            ok = 1
        res[unsafe_offset=i] = ok


@export("ae_flag_high_bit")
def ae_flag_high_bit(vals_addr: Int, n: Int, res_addr: Int) abi("C"):
    """`aenum._enum._high_bit`: index of the highest set bit, -1 for zero.

    This is the index the flag `_generate_next_value_` doubles to get the next
    power of two, so it has to agree with `int.bit_length() - 1` exactly.
    """
    var vals = ip(vals_addr)
    var res = ip(res_addr)
    for i in range(n):
        var v = vals[unsafe_offset=i]
        res[unsafe_offset=i] = Int64(highest_bit(v))


@export("ae_flag_split")
def ae_flag_split(vals_addr: Int, mask: Int, n: Int, res_addr: Int,
                  counts_addr: Int, stride: Int) abi("C"):
    """`aenum._iter_bits_lsb` over a whole array, LSB first.

    Each value is first masked, which is how `Flag.__iter__` restricts itself
    to the single-bit members of a class (`value & cls._singles_mask_`). The
    same lowest-set-bit trick as upstream is used, so the output order is the
    order aenum yields. `res` is n x stride, row major; `counts` holds how
    many bits each row produced. Inputs are non-negative: upstream's loop does
    not terminate on a negative value, and the shim rejects those rather than
    reproducing the hang.
    """
    var vals = ip(vals_addr)
    var res = ip(res_addr)
    var counts = ip(counts_addr)
    for i in range(n):
        var v = vals[unsafe_offset=i] & Int64(mask)
        var row = i * stride
        var c = 0
        while v != 0 and c < stride:
            var bit = v & (~v + 1)
            res[unsafe_offset=row + c] = bit
            v = v ^ bit
            c += 1
        counts[unsafe_offset=i] = Int64(c)


@export("ae_flag_contains")
def ae_flag_contains(vals_addr: Int, other: Int, n: Int, res_addr: Int) abi("C"):
    """`Flag.__contains__`: True when `other`'s bits are all present in value.

    Upstream short-circuits to False when either side is zero, so a zero
    `other` is *not* contained in anything; that quirk is part of the
    contract and is reproduced here.
    """
    var vals = ip(vals_addr)
    var res = ip(res_addr)
    var other64 = Int64(other)
    for i in range(n):
        var v = vals[unsafe_offset=i]
        var ok = Int64(0)
        if other64 != 0 and v != 0 and (other64 & v) == other64:
            ok = 1
        res[unsafe_offset=i] = ok


@export("ae_flag_next_value")
def ae_flag_next_value(last_addr: Int, count: Int, n: Int,
                       res_addr: Int) abi("C"):
    """`Flag._generate_next_value_`: the next power of two above the last value.

    For the first member (count == 0) the value is 1, or `start` when the
    caller gave one; afterwards it is `2 ** (high_bit(max(last_values)) + 1)`,
    which is what keeps a flag from reusing a bit.
    """
    var last = ip(last_addr)
    var res = ip(res_addr)
    for i in range(n):
        if count == 0:
            res[unsafe_offset=i] = Int64(1)
        else:
            var v = last[unsafe_offset=i]
            if v == 0:
                res[unsafe_offset=i] = Int64(1)
            else:
                # 2 ** (high_bit(last) + 1): the power of two above the last
                res[unsafe_offset=i] = Int64(1) << Int64(highest_bit(v) + 1)
