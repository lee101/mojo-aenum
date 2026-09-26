"""Mojo port of aenum's flag algebra.

aenum (Python Advanced Enumerations & NameTuples) is a type-system library:
3,382 lines of enum construction, member collection, `auto()` resolution,
named tuples and descriptors. There is no array math in it anywhere. The one
genuinely numeric surface is the integer bit algebra its `Flag` type is built
from, and that is what is ported here — as batch kernels over arrays of flag
values rather than one member at a time.

```python
import mojo_aenum

mojo_aenum.split([0b1011])          # [[1, 2, 8]] - aenum.split(11)
mojo_aenum.combine([0b1100], 0b1010, "or")   # array([14])
mojo_aenum.contains([0b1111, 0b1000], 0b0010)  # array([ True, False])
```

Everything else in aenum — `Enum`, `IntEnum`, `AutoNumberEnum`, `AddValueEnum`,
`NamedConstant`, `enum_property`, `extend_enum`, `add_stdlib_integration`,
the whole `auto()` protocol — is control flow and stays in the real package.
"""

from .flags import (
    combine,
    contains,
    flag_len,
    high_bit,
    is_single_bit,
    iter_flag_values,
    next_flag_value,
    split,
)

__all__ = [
    "combine",
    "contains",
    "flag_len",
    "high_bit",
    "is_single_bit",
    "iter_flag_values",
    "next_flag_value",
    "split",
]
__version__ = "0.1.0"
