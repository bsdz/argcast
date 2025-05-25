# argcast - Run time function argument caster

Automatically cast function parameters at run time based on their types.


## Installation

```bash
pip install argcast
```

## Usage

Below is an example usage:

```python
from decimal import Decimal
from enum import Enum

import numpy as np
import pandas as pd

from argcast import coerce_params

class MatrixOp(Enum):
    INVERSE = 1
    TRANSPOSE = 2

    @classmethod
    def get(cls, name):
        return cls[name] if isinstance(name, str) else cls(name)

coerce = coerce_params({np.ndarray: np.array, MatrixOp: MatrixOp.get})

@coerce
def f(
    a: np.ndarray, b: np.ndarray, k: np.float64, b_trans: MatrixOp
) -> pd.DataFrame:

    if b_trans == MatrixOp.INVERSE:
        b = np.linalg.inv(b)
    elif b_trans == MatrixOp.TRANSPOSE:
        b = b.T
    return k * a @ b

f([[1, 2], [3, 4]], [[5, 6], [7, 8]], Decimal("2.0"), "TRANSPOSE")
# returns pd.DataFrame([[34.0, 46.0], [78.0, 106.0]])
```

## Limitations

1) Using this decorator will confuse type checkers like mypy and you
might wish to place a top level comment in your module like:

```python
# mypy: disable-error-code="return-value"
```

2) Although basic sequence and mapping types are supported, ie list|tuple|set[str|int|...]
or dict[..., ...], many other types aren't (eg nested types list[list[...]] etc).

## Development

Before commit run following format commands in project folder:

```bash
poetry run black .
poetry run isort . --profile black
poetry run docformatter . --recursive --in-place --black
```
