"""
Convenient JAX typings.
"""
import jax
import numpy as np
from typing import Union

# The three types are exactly the same alias of jax.Array. We differ them only semantically.
JArray = jax.Array
JInt = jax.Array
JFloat = jax.Array
JBool = jax.Array
JKey = jax.Array

# Arrays
Array = Union[JArray, np.ndarray]

# Scalar values
# The syntax float | DeviceFloat is not supported before Python 3.10
FloatScalar = Union[float, JFloat]
IntScalar = Union[int, JFloat]
BoolScalar = Union[bool, JBool]
