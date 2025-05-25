from functools import wraps
from inspect import signature
from types import GenericAlias


class DoNotCoerce:
    """Marker class to indicate that a parameter should not be coerced."""

    pass


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
            type_map.update(arg)

    def _get_type(g, an, tm):
        g_arg_ann = g.__annotations__.get(an)
        if isinstance(g_arg_ann, GenericAlias):
            # TODO: ga_orig should be mapped to new type from type_map
            ga_orig, ga_args = g_arg_ann.__origin__, g_arg_ann.__args__
            if ga_orig in (list, tuple, set) and len(ga_args) == 1:
                return lambda x: ga_orig([tm.get(ga_args[0], ga_args[0])(i) for i in x])
            elif ga_orig in (dict,) and len(ga_args) == 2:
                return lambda x: ga_orig(
                    {
                        tm.get(ga_args[0], ga_args[0])(k): tm.get(
                            ga_args[1], ga_args[1]
                        )(v)
                        for k, v in (x.items() if isinstance(x, dict) else x)
                    }
                )
        return (tm.get(g_arg_ann)) or g_arg_ann

    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            ba = signature(f).bind(*args, **kwargs)

            for kw, val in ba.arguments.items():
                am_type = arg_map.get(kw)
                if am_type is DoNotCoerce:
                    continue
                if typ := am_type or _get_type(f, kw, type_map):
                    ba.arguments[kw] = typ(val)

            val = f(*ba.args, **ba.kwargs)

            if typ := _get_type(f, "return", type_map):
                val = typ(val)

            return val

        return wrapper

    return wrap if func is None else wrap(func)
