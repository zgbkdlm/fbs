import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from flax import linen
from fbs.typings import JArray, JKey, FloatScalar
from typing import Tuple, Callable


def make_nn_with_time(nn: linen.Module,
                      dim_in: int,
                      batch_size: int,
                      time_scale: FloatScalar,
                      key: JKey) -> Tuple[JArray, Callable[[JArray], dict], Callable[[JArray, JArray], JArray]]:
    """Make a neural network with time embedding (the baby version).

    Parameters
    ----------
    nn : linen.Module
        A neural network instance.
    dim_in : int
        The input dimension.
    batch_size : int
        The data batch size.
    time_scale : FloatScalar
        Scale the time variable to keep consistent with the norm of the input.
    key : JKey
        A JAX random key.

    Returns
    -------
    JArray, Callable[[JArray], dict], Callable (d, ), (p, ) -> (d, )
        The initial parameter array, the array-to-dict ravel function, and the NN forward pass evaluation function.
    """
    dict_param = nn.init(key, jnp.ones((batch_size, dim_in + 1)))
    array_param, array_to_dict = ravel_pytree(dict_param)

    def forward_pass(x: JArray, t: FloatScalar, param: JArray) -> JArray:
        """The NN forward pass.
        x : (d, )
        t : Any float number
        param : (p, )
        return : (d, )
        """
        return nn.apply(array_to_dict(param), jnp.hstack([x, t * time_scale]))

    return array_param, array_to_dict, forward_pass


def sinusoidal_embedding():
    # TODO: implement sinusoidal embedding as in the following link
    # https://github.com/JTT94/diffusion_schrodinger_bridge/blob/1c82eba0a16aea3333ac738dde376b12a3f97f21/bridge/models/basic/time_embedding.py#L6
    pass
