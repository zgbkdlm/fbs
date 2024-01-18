import jax.numpy as jnp
import einops
from jax.flatten_util import ravel_pytree
from flax import linen
from fbs.typings import JArray, JKey, FloatScalar
from typing import Tuple, Callable


def make_nn_with_time(nn: linen.Module,
                      dim_in: int,
                      batch_size: int,
                      time_scale: FloatScalar,
                      key: JKey) -> Tuple[
    JArray, Callable[[JArray], dict], Callable[[JArray, FloatScalar, JArray], JArray]]:
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


class PixelShuffle(linen.Module):
    scale: int

    def __call__(self, x: JArray) -> JArray:
        return einops.rearrange(x, 'b h w (h2 w2 c) -> b (h h2) (w w2) c', h2=self.scale, w2=self.scale)
