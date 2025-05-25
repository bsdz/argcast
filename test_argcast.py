# mypy: disable-error-code="return-value"

import unittest
from decimal import Decimal
from enum import Enum

from argcast import DoNotCoerce
from argcast import coerce_params as coerce

HAS_NUMPY = True
try:
    import numpy as np
    import numpy.typing as npt
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

        coerce = coerce_params({npt.ArrayLike: np.asarray, MatrixOp: MatrixOp.get})

        @coerce
        def f(
            a: npt.ArrayLike, b: npt.ArrayLike, k: np.float64, b_trans: MatrixOp
        ) -> pd.DataFrame:

            if b_trans == MatrixOp.INVERSE:
                b = np.linalg.inv(b)  # type: ignore[arg-type]
            elif b_trans == MatrixOp.TRANSPOSE:
                b = b.T  # type: ignore[union-attr]
            return k * a @ b  # type: ignore[union-attr,call-overload,operator]

        self.assertTrue(
            f([[1, 2], [3, 4]], [[5, 6], [7, 8]], Decimal("2.0"), "TRANSPOSE").equals(
                pd.DataFrame([[34.0, 46.0], [78.0, 106.0]])
            )
        )

    def test_sequences(self):

        @coerce
        def f(a: list[str], b: list[str]) -> str:
            return "".join(a) + "".join(b)

        self.assertEqual(f([1, 2, 3], [4, 5, 6]), "123456")

        @coerce
        def g(a: list[int], b: list[int]) -> tuple[str, ...]:
            return a + b

        self.assertEqual(g([1, 2, 3], ["4", "5", "6"]), (1, 2, 3, 4, 5, 6))

        @coerce
        def h(a: set[int], b: set[int]) -> tuple[str, ...]:
            return a | b

        self.assertEqual(h(tuple([1, 2, 3]), ["3", "4", "5"]), (1, 2, 3, 4, 5))

    def test_mappings(self):

        @coerce
        def f(a: dict[str, str], b: dict[str, str]) -> dict[str, int]:
            return list((a | b).items())

        self.assertEqual(
            f([("a", 1), ("b", 2)], {"a": 3, "b": 4, "c": 5}), {"a": 3, "b": 4, "c": 5}
        )

    def test_override(self):
        @coerce(a=int, b=float)
        def f(a, b):
            return a + b

        self.assertEqual(f("1", "2.0"), 3.0)

        @coerce(a=DoNotCoerce)
        def g(a: int, b: int) -> str:
            return a + b

        self.assertEqual(g(1, "2"), "3")

        with self.assertRaises(TypeError):
            g("1", "2")


if __name__ == "__main__":
    unittest.main()
