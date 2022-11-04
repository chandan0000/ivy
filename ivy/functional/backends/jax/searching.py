from numbers import Number
from typing import Optional, Tuple, Union

import jax.numpy as jnp

import ivy
from ivy.functional.backends.jax import JaxArray


# Array API Standard #
# ------------------ #


def argmax(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ret = jnp.argmax(x, axis=axis, keepdims=keepdims)
    if output_dtype:
        ret = ret.astype(output_dtype)
    return ret


def argmin(
    x: JaxArray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[jnp.dtype] = jnp.int64,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    ret = jnp.argmin(x, axis=axis, keepdims=keepdims)
    if dtype is not None:
        return (
            jnp.array(ret, dtype=jnp.int32)
            if dtype not in (jnp.int32, jnp.int64)
            else jnp.array(ret, dtype=dtype)
        )

    if ret.dtype not in (jnp.int32, jnp.int64):
        return jnp.array(ret, dtype=jnp.int32)
    else:
        return jnp.array(ret, dtype=ret.dtype)


def nonzero(
    x: JaxArray,
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[JaxArray, Tuple[JaxArray]]:
    res = jnp.nonzero(x, size=size, fill_value=fill_value)

    return tuple(res) if as_tuple else jnp.stack(res, axis=1)


def where(
    condition: JaxArray,
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ivy.astype(jnp.where(condition, x1, x2), x1.dtype, copy=False)


# Extra #
# ----- #


def argwhere(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.argwhere(x)
