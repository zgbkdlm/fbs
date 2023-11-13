import jax
import jax.numpy as jnp
from fbs.typings import JArray, JKey, JFloat, FloatScalar
from typing import Callable, Tuple


def ipf_fwd_loss(f_param: JArray,
                 b_param: JArray,
                 f: Callable[[JArray, FloatScalar, JArray], JArray],
                 b: Callable[[JArray, FloatScalar, JArray], JArray],
                 x0s: JArray,
                 ts: JArray,
                 key: JKey) -> JFloat:
    """Mean-matching iterative proportional fitting (the forward loss).

    Algorithm 1. of de Bortoli et al., 2021.

    Parameters
    ----------
    f_param : JArray (p, )
        An array of parameters for the function `f`.
    b_param : JArray (p, )
        An array of parameters for the function `b`.
    f : Callable (..., d), (), (p, ) -> (..., d)
        A parametric function for the forward process, taking three arguments: state, time, and parameters.
    b : Callable (..., d), (), (p, ) -> (..., d)
        A parametric function for the backward process, taking three arguments: state, time, and parameters.
    x0s : JArray (n, d)
        The samples at the initial side.
    ts : JArray (nsteps + 1, )
        Times, including t0 and tT.
    key : JKey
        A JAX random key.

    Returns
    -------
    JArray (p, ), JArray (p, )
        Two arrays carrying the parameters for `f` and `b`.

    Notes
    -----
    Double-check the annoying square two in the dispersion.
    """
    dts = jnp.diff(ts)
    n, d = x0s.shape

    def init_scan(carry, elem):
        x, err = carry
        t, t_next, dw = elem

        x_next = f(x, t, f_param) + dw
        err = err + jnp.sum((b(x_next, t_next, b_param) - (x_next + f(x, t, f_param) - f(x_next, t, f_param))) ** 2)
        return (x, err), None

    key, subkey = jax.random.split(key)
    dws = jnp.sqrt(dts) * jax.random.normal(subkey, (dts.shape[0], d))
    errs = jax.vmap(lambda x0: jax.lax.scan(init_scan, (x0, 0.), (ts[:-1], ts[1:], dws))[0][1], in_axes=[0])(x0s)
    return jnp.mean(errs)


def ipf_bwd_loss(b_param: JArray,
                 f_param: JArray,
                 b: Callable[[JArray, FloatScalar, JArray], JArray],
                 f: Callable[[JArray, FloatScalar, JArray], JArray],
                 xTs: JArray,
                 ts: JArray,
                 key: JKey) -> JFloat:
    """Mean-matching iterative proportional fitting (the backward loss).

    See the docstring of `ipf_fwd_loss`. Note that `ts = (t_T, t_{T-1}, ..., t_0) = T - ts = ts[::-1]`
    """
    dts = jnp.diff(ts)
    n, d = xTs.shape

    def init_scan(carry, elem):
        x, err = carry
        t, t_next, dw = elem

        x_next = b(x, t, f_param) + dw
        err = err + jnp.sum((f(x_next, t_next, f_param) - (x_next + b(x, t, b_param) - b(x_next, t, b_param))) ** 2)
        return (x, err), None

    key, subkey = jax.random.split(key)
    dws = jnp.sqrt(dts) * jax.random.normal(subkey, (dts.shape[0], d))
    errs = jax.vmap(lambda xT: jax.lax.scan(init_scan, (xT, 0.), (ts[:-1], ts[1:], dws))[0][1], in_axes=[0])(xTs)
    return jnp.mean(errs)
