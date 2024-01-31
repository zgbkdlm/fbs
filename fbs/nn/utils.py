import jax
import jax.numpy as jnp
import optax
import einops
from jax.flatten_util import ravel_pytree
from flax import linen
from fbs.typings import JArray, JKey, FloatScalar
from functools import partial
from typing import Tuple, Callable, Sequence


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


def make_optax_kernel(optimiser, loss_fn: Callable, jit: bool = True) -> Tuple[Callable, Callable]:
    def optax_kernel(param: JArray, opt_state, *args, **kwargs):
        loss, grad = jax.value_and_grad(loss_fn)(param, *args, **kwargs)
        updates, opt_state = optimiser.update(grad, opt_state, param)
        param = optax.apply_updates(param, updates)
        return param, opt_state, loss

    @partial(jax.jit, static_argnums=2)
    def ema_update(param: JArray, ema_param: JArray, decay: float) -> JArray:
        return decay * ema_param + (1 - decay) * param

    def ema_kernel(ema_param: JArray, param: JArray, count: int, count_threshold: int, decay: float) -> JArray:
        if count < count_threshold:
            ema_param = param
        else:
            ema_param = ema_update(param, ema_param, decay)
        return ema_param

    return jax.jit(optax_kernel) if jit else optax_kernel, ema_kernel
