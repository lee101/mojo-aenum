# mojo-aenum

Mojo port of the compute surface of [aenum](https://github.com/CyrilPepin/aenum)
(Python Advanced Enumerations & NameTuples), version 3.1.17. The Python
package is named `mojo_aenum`, so it installs alongside the real `aenum` and
the parity tests import both and compare them directly.

## Does aenum have a compute core?

No, and this README will not pretend otherwise. aenum is a type-system
library. Its 3,382 lines of `_enum.py` are enum construction, member
collection, `auto()` resolution, `AddValue`/`MultiValue`/`NoAlias` policies,
named tuples, descriptors and aliases. There is not a single array operation
in the package, and no loop over anything larger than one enum class.

What aenum *does* compute is integer bit algebra, because `Flag` is defined by
it. That is the whole numeric surface of the package, and it is what is ported:

| aenum operation | expression | ported as |
| --- | --- | --- |
| `Flag.__or__` / `__and__` / `__xor__` | `value \| other`, `value & other`, `value ^ other` | `combine(values, mask, "or"/"and"/"xor")` |
| `Flag.__invert__` | `~value` | `combine(values, op="invert")`, and `combine(values, mask, "invert_masked")` for aenum's width-limited form |
| `Flag.__contains__` | `other & value == other`, false if either side is zero | `contains(values, other)` |
| `Flag.__len__` | `bit_count(value)` (Kernighan) | `flag_len(values)` |
| `_iter_bits_lsb` / `show_flag_values` | `bit = value & (~value + 1); value ^= bit` | `split(values, mask=None)` |
| `Flag.__iter__` | `_iter_bits_lsb(value & cls._singles_mask_)` | `iter_flag_values(values, singles_mask)` |
| `is_single_bit` | `num != 0 and num & (num - 1) == 0` | `is_single_bit(values)` |
| `_high_bit` | `value.bit_length() - 1` | `high_bit(values)` |
| `Flag._generate_next_value_` | `2 ** (_high_bit(max(values)) + 1)` | `next_flag_value(last_values, count)` |

The widening is deliberate and is the only liberty taken: aenum evaluates each
of these on a single member, through a Python `int`, one call at a time. These
kernels evaluate the identical integer expressions over a whole array of flag
values, which is the only form in which they are worth compiling.

Everything is exact integer arithmetic, so no tolerance is involved anywhere:
the parity tests assert `==`, not `assert_allclose`.

## Not implemented

- All of aenum's type machinery: `Enum`, `IntEnum`, `AutoNumberEnum`,
  `OrderedEnum`, `UniqueEnum`, `StrEnum`, `AddValueEnum`, `MultiValueEnum`,
  `NoAliasEnum`, `NamedConstant`, `Constant`, `enum_property`,
  `extend_enum`, `add_stdlib_integration`, `NamedTuple`, and the whole `auto()`
  protocol. Use the real package for those.
- `aenum.bin` and `aenum.bits`, which render a value as a formatted bit
  string. That is string formatting, not arithmetic.
- Flag values outside a signed 64-bit word. aenum's Python ints are unbounded;
  the kernels are `int64`. `high_bit` refuses values above 2**63, and
  `next_flag_value` refuses to answer when the next power of two would not fit
  (2**63, 2**64) instead of wrapping to a negative number.
- Negative flag values for `split`. Upstream's `_iter_bits_lsb` does not
  terminate on a negative value at all, because `value ^= bit` leaves a
  negative value negative; the shim raises `ValueError` rather than
  reproducing the hang.

One semantic note worth stating plainly: aenum's `~flag` is *not* the raw
64-bit complement. `Flag.__invert__` builds `self.__class__(~self._value_)`,
and the class constructor applies the class's boundary policy, so the result is
the complement inside the class's own mask (and, for a `KEEP`-boundary
`IntFlag` holding bits outside that mask, inside `max(bit_length(v), mask
bits)`). `combine(..., "invert_masked")` with the class mask reproduces that
for values inside the mask; `combine(..., "invert")` is the raw complement.

## Install

```bash
pixi install
pixi run build     # -> dist/libmojo-aenum.so
pixi run test
pixi run bench
```

`PYTHONPATH=python` is set by the Pixi activation. On the shared toolchain
box, never run `pixi install`: use the shared environment and
`source /nvme0n1-disk/mojo-toolchain/activate.sh`.

## Usage

```python
import mojo_aenum

mojo_aenum.split([0b1011, 0b1100])          # [[1, 2, 8], [4, 8]]
mojo_aenum.combine([0b1100], 0b1010, "or")   # array([14])
mojo_aenum.flag_len([0b1011])               # array([3])
mojo_aenum.contains([0b1111, 0b1000], 2)    # array([ True, False])
mojo_aenum.next_flag_value([1, 2, 4], 1)    # array([2, 4, 8])
```

## Tests

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

197 tests, all against the real aenum: `aenum._enum.show_flag_values` (the
public form of `_iter_bits_lsb`), `is_single_bit`, `_high_bit`, `bit_len`,
and the operators, `__len__`, `__contains__` and `__iter__` of a real `IntFlag`
subclass. They cover the bit-width boundaries (2**20, 2**31, 2**61, 2**62),
single-bit and multi-bit values, zero, empty batches, input immutability, and
the out-of-range and negative-input refusals described above.

## Performance

Best-of-five, same process, n = 4,194,304, against vectorised NumPy
(`|`, `&`, `^`, `~` are NumPy's own; the population count is a SWAR popcount;
the highest-set-bit baseline is `np.frexp`). Every case verifies exact equality
before timing. The box is shared, so the absolute numbers move a few tens of
percent between runs.

| case | baseline | mojo-aenum | result |
| --- | ---: | ---: | ---: |
| combine or | 65.18 ms | 28.91 ms | 2.25x faster |
| combine and | 19.76 ms | 18.19 ms | 1.09x faster |
| combine xor | 22.22 ms | 24.30 ms | 0.91x, parity |
| combine invert | 19.85 ms | 22.98 ms | 0.86x, parity |
| contains | 29.59 ms | 49.80 ms | 0.59x, slower |
| flag_len (popcount) | 193.30 ms | 19.73 ms | 9.80x faster |
| high_bit | 89.48 ms | 123.39 ms | 0.73x, slower |
| split (vs aenum's generator) | 1521.75 ms | 641.40 ms | 2.37x faster |

The two wins are the two kernels that do real work per element: the Kernighan
popcount loop is 9.8x faster than SWAR popcount on NumPy because it is a
sequential dependency chain in the SWAR formulation, and `split` beats a
Python generator by 2.4x because the bit extraction is in compiled code
instead of a generator frame. The plain bitwise cases are memory-bound and
NumPy is already at memory speed, so parity is the honest answer there.
`contains` is slower than NumPy because the faithful version has to apply
aenum's zero short circuit, which NumPy's `(v & m) == m` does not; the
baseline it is compared against is therefore slightly under-specified, which is
why the comparison is reported rather than dressed up. `high_bit` loses to
`np.frexp`, which is a hardware exponent extraction rather than a shift
search.

## How it works

`src/kernels.mojo` is one compilation unit containing the six exported
kernels; `build/build.sh` compiles it to `dist/libmojo-aenum.so`. Buffers cross
the C ABI as 64-bit addresses and are rebuilt in Mojo as
`Pointer[Int64, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric.

The split kernel is the only one with structure worth describing: it uses
upstream's own lowest-set-bit trick, so the bits come out in exactly the order
`Flag.__iter__` yields them, and the output is a fixed-stride matrix plus a
count array, because a ragged Python list per value would move the cost back
into the interpreter.

## License

MIT
