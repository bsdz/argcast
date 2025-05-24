from functools import wraps
from inspect import signature


def coerce_params(*args, **arg_map):
    """Automatically cast function arguments to the specified types. This decorator
    allows you to specify the types of function arguments and the return type. It will
    automatically cast the arguments to the specified types before calling the function.

    Examples:

    Simple usage:

        >>> @coerce
        >>> def f1(a: int, b: float) -> str:
        >>>     return a + b

        >>> f1(1, 2.0)
        "3.0"

    With type mapping:

        >>> @coerce({np.ndarray: np.array})
        >>> def f3(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
        >>>     return a @ b

        >>> f3([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        DataFrame([[19, 22], [43, 50]])

    With argument type mapping:

        >>> @coerce(a=int, b=float)
        >>> def f2(a, b):
        >>>     return a + b

        >>> f2(1, 2.0)
        "3.0"
    """

    func = None
    type_map = {}

    for arg in args:
        if callable(arg):
            func = arg
        elif isinstance(arg, dict):
            type_map = arg

    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ba = signature(f).bind(*args, **kwargs)

            for kw, val in ba.arguments.items():
                if (
                    typ := arg_map.get(kw)
                    or (type_map.get(f.__annotations__.get(kw)))
                    or f.__annotations__.get(kw)
                ):
                    ba.arguments[kw] = typ(val)

            val = f(*ba.args, **ba.kwargs)

            if typ := (
                type_map.get(f.__annotations__.get("return"))
            ) or f.__annotations__.get("return"):
                val = typ(val)

            return val

        return wrapper

    return wrap if func is None else wrap(func)
