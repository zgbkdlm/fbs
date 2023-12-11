import math
import flax.linen as linen
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from fbs.typings import JArray, FloatScalar, JKey
from typing import Union, Tuple, Callable


def make_st_nn(nn: linen.Module,
               dim_x: int,
               batch_size: int,
               key: JKey) -> Tuple[
    JArray, Callable[[JArray], dict], Callable[[JArray, FloatScalar, JArray], JArray]]:
    """Make a neural network for approximating a spatial-temporal function :math:`f(x, t)`.

    Parameters
    ----------
    nn : linen.Module
        A neural network instance.
    dim_x : int
        The spatial dimension.
    batch_size : int
        The data batch size.
    key : JKey
        A JAX random key.

    Returns
    -------
    JArray, Callable[[JArray], dict], Callable (..., d), (p, ) -> (..., d)
        The initial parameter array, the array-to-dict ravel function, and the NN forward pass evaluation function.
    """
    dict_param = nn.init(key, jnp.ones((batch_size, dim_x)), jnp.ones((batch_size, 1)))
    array_param, array_to_dict = ravel_pytree(dict_param)

    def forward_pass(x: JArray, t: FloatScalar, param: JArray) -> JArray:
        """The NN forward pass.
        x : (..., d)
        t : (...)
        param : (p, )
        return : (..., d)
        """
        return nn.apply(array_to_dict(param), x, t)

    return array_param, array_to_dict, forward_pass


def sinusoidal_embedding(t: Union[JArray, FloatScalar], out_dim: int = 64, max_period: int = 10_000) -> JArray:
    """The so-called sinusoidal positional embedding.

    Parameters
    ----------
    t : FloatScalar or JArray (...)
        A time variable or batched severals.
    out_dim : int
        The output dimension.
    max_period : int
        The maximum period.

    Returns
    -------
    JArray (..., out_dim)
        An array.

    Notes
    -----
    I have no idea what a sinusoidal positional embedding is. Perhaps, it means to find a function that maps a time
    scalar to a sequence. The implementation is based on
        - https://github.com/JTT94/diffusion_schrodinger_bridge/blob/1c82eba0a16aea3333ac738dde376b12a3f97f21/
        bridge/models/basic/time_embedding.py#L6
        - https://github.com/vdeborto/cdsb/blob/8fc9cc2a08daa083b84b5ddd38190bec931edeb0/
        bridge/models/unet/layers.py#L95
    """
    half = out_dim // 2

    fs = jnp.exp(-math.log(max_period) * jnp.arange(half) / (half - 1))
    embs = t * fs
    embs = jnp.concatenate([jnp.sin(embs), jnp.cos(embs)], axis=-1)
    if out_dim % 2 == 1:
        raise NotImplementedError(f'out_dim is implemented for even number only, while {out_dim} is given.')
    return embs
