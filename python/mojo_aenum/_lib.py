"""ctypes bridge to the compiled Mojo flag-algebra kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a
64-bit address, so the argtypes below must stay `c_int64` for addresses;
`c_int` truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-aenum.so"

OP_OR = 0
OP_AND = 1
OP_XOR = 2
OP_INVERT = 3
OP_INVERT_MASKED = 4

_OP_NAMES = {
    "or": OP_OR,
    "|": OP_OR,
    "and": OP_AND,
    "&": OP_AND,
    "xor": OP_XOR,
    "^": OP_XOR,
    "invert": OP_INVERT,
    "~": OP_INVERT,
    "invert_masked": OP_INVERT_MASKED,
}

WORD_BITS = 64


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ae_flag_combine.restype = None
    lib.ae_flag_combine.argtypes = [ctypes.c_int64] * 5
    lib.ae_flag_popcount.restype = None
    lib.ae_flag_popcount.argtypes = [ctypes.c_int64] * 3
    lib.ae_flag_single_bit.restype = None
    lib.ae_flag_single_bit.argtypes = [ctypes.c_int64] * 3
    lib.ae_flag_high_bit.restype = None
    lib.ae_flag_high_bit.argtypes = [ctypes.c_int64] * 3
    lib.ae_flag_split.argtypes = [ctypes.c_int64] * 6
    lib.ae_flag_split.restype = None
    lib.ae_flag_contains.restype = None
    lib.ae_flag_contains.argtypes = [ctypes.c_int64] * 4
    lib.ae_flag_next_value.restype = None
    lib.ae_flag_next_value.argtypes = [ctypes.c_int64] * 4
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def op_code(op) -> int:
    if op not in _OP_NAMES:
        raise ValueError(
            f"op must be one of {sorted(set(_OP_NAMES))}, got {op!r}"
        )
    return _OP_NAMES[op]


def combine(values, mask=0, op="or"):
    """Elementwise `v | mask`, `v & mask`, `v ^ mask`, `~v` or `~v & mask`."""
    vals = np.ascontiguousarray(values, dtype=np.int64)
    res = np.empty(vals.size, dtype=np.int64)
    lib.ae_flag_combine(
        _addr(vals), int(mask), vals.size, _addr(res), op_code(op)
    )
    return res


def popcount(values):
    """`Flag.__len__` per value: how many bits are set."""
    vals = np.ascontiguousarray(values, dtype=np.int64)
    res = np.empty(vals.size, dtype=np.int64)
    lib.ae_flag_popcount(_addr(vals), vals.size, _addr(res))
    return res


def single_bit(values):
    """`aenum.is_single_bit` per value."""
    vals = np.ascontiguousarray(values, dtype=np.int64)
    res = np.empty(vals.size, dtype=np.int64)
    lib.ae_flag_single_bit(_addr(vals), vals.size, _addr(res))
    return res.astype(bool)


def high_bit(values):
    """`aenum._enum._high_bit` per value: -1 when the value is zero."""
    vals = np.ascontiguousarray(values, dtype=np.int64)
    res = np.empty(vals.size, dtype=np.int64)
    lib.ae_flag_high_bit(_addr(vals), vals.size, _addr(res))
    return res


def split(values, mask=None, stride=None):
    """`aenum._iter_bits_lsb` per value, masked, LSB first.

    Returns `(bits, counts)` where `bits` is an (n, stride) matrix and
    `counts[i]` is how many bits row i produced.
    """
    vals = np.ascontiguousarray(values, dtype=np.int64)
    if (vals < 0).any():
        # aenum's own loop never terminates on a negative value, because
        # `value ^= bit` leaves a negative value negative forever.
        raise ValueError("flag values must be non-negative")
    m = np.iinfo(np.int64).max if mask is None else int(mask)
    if m < 0:
        raise ValueError("the mask must be non-negative")
    st = WORD_BITS if stride is None else int(stride)
    if st < 1 or st > WORD_BITS:
        raise ValueError(f"stride must be in 1..{WORD_BITS}")
    res = np.zeros((vals.size, st), dtype=np.int64)
    counts = np.empty(vals.size, dtype=np.int64)
    lib.ae_flag_split(
        _addr(vals), m, vals.size, _addr(res), _addr(counts), st
    )
    return res, counts


def contains(values, other):
    """`Flag.__contains__` per value."""
    vals = np.ascontiguousarray(values, dtype=np.int64)
    res = np.empty(vals.size, dtype=np.int64)
    lib.ae_flag_contains(_addr(vals), int(other), vals.size, _addr(res))
    return res.astype(bool)


def next_flag_value(last_values, count):
    """`Flag._generate_next_value_`: the next free power of two.

    aenum's integers are unbounded, so `2 ** 64` is a legal aenum value. The
    kernel works in int64, so the shim refuses the one input whose answer
    does not fit instead of wrapping.
    """
    vals = np.ascontiguousarray(last_values, dtype=np.int64)
    res = np.empty(vals.size, dtype=np.int64)
    lib.ae_flag_next_value(_addr(vals), int(count), vals.size, _addr(res))
    if ((vals > 0) & (res <= 0)).any():
        raise OverflowError(
            "the next flag value does not fit in a signed 64-bit word"
        )
    return res
