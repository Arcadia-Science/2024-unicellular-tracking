from functools import wraps
from time import time
import numpy as np


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


def setdiff2d(a, b):
    """Finding the difference between two 2D arrays is surprisingly involved...
    
    References
    ----------
    [1] https://stackoverflow.com/a/11903368/5285918
    """

    a_ = a.view([('', a.dtype)] * a.shape[1])
    b_ = b.view([('', b.dtype)] * b.shape[1])
    c = np.setdiff1d(a_, b_).view(a.dtype).reshape(-1, a.shape[1])

    return c
