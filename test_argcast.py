# mypy: disable-error-code="return-value"

import unittest
from decimal import Decimal
from enum import Enum

from argcast import coerce_params as coerce

HAS_NUMPY = True
try:
    import numpy as np
except ImportError:
    HAS_NUMPY = False

HAS_PANDAS = True
try:
    import pandas as pd
except ImportError:
    HAS_PANDAS = False


class TestAutocast(unittest.TestCase):
    def test_simple(self):

        @coerce
        def f(a: str, b: int) -> str:
            return a * b

        self.assertEqual(f(0, 2), "00")

        @coerce
        def g(a: int, b: int) -> str:
            return a * b

        self.assertEqual(g("2", 3.2), "6")

        @coerce(b=lambda x: int(x + 2))
        def h(a: int, b: int) -> str:
            return a * b

        self.assertEqual(h(2, 3), "10")
        self.assertEqual(h(2, 3.0), "10")

    def test_enum(self):

        class MyEnum(Enum):
            FOO = 1
            BAR = 2

            @classmethod
            def get(cls, name):
                return cls[name] if isinstance(name, str) else cls(name)

        @coerce(a=MyEnum.get)
        def f(a: MyEnum, b: int) -> str:
            if a == MyEnum.FOO:
                return b * 2
            return b * 3

        self.assertEqual(f("FOO", 5), "10")

    def test_decimal(self):

        @coerce
        def f(a: Decimal, b: Decimal) -> float:
            return a * b

        self.assertAlmostEqual(f(10.1, 2.3), 23.23)

        @coerce
        def g(a: Decimal, b: Decimal) -> Decimal:
            return a * b

        self.assertEqual(g(10.1, 2.3), Decimal("23.22999999999999738875544608"))

    @unittest.skipUnless(HAS_NUMPY, "Numpy is not installed")
    def test_pandas(self):
        @coerce
        def f(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
            return a @ b

        self.assertTrue(
            f([[1, 2], [3, 4]], [[5, 6], [7, 8]]).equals(
                pd.DataFrame([[19, 22], [43, 50]])
            )
        )

    @unittest.skipUnless(
        HAS_NUMPY and HAS_PANDAS, "Numpy and/or Pandas is not installed"
    )
    def test_type_map(self):
        @coerce({np.ndarray: np.array})
        def f(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
            return a @ b

        self.assertTrue(
            f([[1, 2], [3, 4]], [[5, 6], [7, 8]]).equals(
                pd.DataFrame([[19, 22], [43, 50]])
            )
        )

        mycoerce = coerce({np.ndarray: np.array})

        @mycoerce
        def g(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
            return a @ b

        self.assertTrue(
            g([[1, 2], [3, 4]], [[5, 6], [7, 8]]).equals(
                pd.DataFrame([[19, 22], [43, 50]])
            )
        )

    @unittest.skipUnless(
        HAS_NUMPY and HAS_PANDAS, "Numpy and/or Pandas is not installed"
    )
    def test_readme_example(self):
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

        self.assertTrue(
            f([[1, 2], [3, 4]], [[5, 6], [7, 8]], Decimal("2.0"), "TRANSPOSE").equals(
                pd.DataFrame([[34.0, 46.0], [78.0, 106.0]])
            )
        )


if __name__ == "__main__":
    unittest.main()
