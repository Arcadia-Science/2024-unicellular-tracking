from functools import wraps
from time import time


def timeit(f):
    """Decorator for outputting the execution time of a function."""

    @wraps(f)
    def wrap(*args, **kwargs):
        t0 = time()
        result = f(*args, **kwargs)
        t1 = time()

        out = f"{f.__name__} :: {t1-t0:.2f}s"
        print(out)
        return result

    return wrap
