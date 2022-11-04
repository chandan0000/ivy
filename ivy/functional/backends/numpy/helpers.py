import functools
from typing import Callable
import numpy as np


# when inputs are 0 dimensional, numpy's functions return scalars
# so we use this wrapper to ensure outputs are always numpy arrays
def _handle_0_dim_output(function: Callable) -> Callable:
    @functools.wraps(function)
    def new_function(*args, **kwargs):
        ret = function(*args, **kwargs)
        return ret if isinstance(ret, np.ndarray) else np.asarray(ret)

    return new_function
