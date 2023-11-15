import jax
import jax.numpy as jnp
from fbs.typings import JArray, JKey, JFloat, FloatScalar
from typing import Callable, Tuple, Any


def ipf_fwd_loss(b_param: JArray,
                 f_param: JArray,
                 f: Callable[[JArray, FloatScalar, JArray], JArray],
                 b: Callable[[JArray, FloatScalar, JArray], JArray],
                 x0s: JArray,
                 ts: JArray,
                 sigma: FloatScalar,
                 key: JKey) -> JFloat:
    """Mean-matching iterative proportional fitting (the forward loss).

    Algorithm 1. of de Bortoli et al., 2021.

    Parameters
    ----------
    b_param : JArray (p, )
        An array of parameters for the function `b`. This is the parameter to be learnt.
    f_param : JArray (p, )
        An array of parameters for the function `f`.
    f : Callable (..., d), (), (p, ) -> (..., d)
        A parametric function for the forward process, taking three arguments: state, time, and parameters.
    b : Callable (..., d), (), (p, ) -> (..., d)
        A parametric function for the backward process, taking three arguments: state, time, and parameters.
    x0s : JArray (n, d)
        The samples at the initial side.
    ts : JArray (nsteps + 1, )
        Times, including t0 and tT.
    sigma : float
        The dispersion term of the process.
    key : JKey
        A JAX random key.

    Returns
    -------
    JFloat
        The loss for the forward objective function.

    Notes
    -----
    Double-check the annoying square two in the dispersion.
    """
    dts = jnp.diff(ts)
    n, d = x0s.shape

    def init_scan(carry, elem):
        x, err = carry
        t, t_next, dw = elem

        x_next = f(x, t, f_param) + sigma * dw
        err = err + jnp.sum((b(x_next, t_next, b_param) - (x_next + f(x, t, f_param) - f(x_next, t, f_param))) ** 2)
        return (x, err), None

    key, subkey = jax.random.split(key)
    dwss = jnp.sqrt(dts)[None, :, None] * jax.random.normal(subkey, (n, dts.shape[0], d))
    errs = jax.vmap(lambda x0, dws: jax.lax.scan(init_scan, (x0, 0.), (ts[:-1], ts[1:], dws))[0][1],
                    in_axes=[0, 0])(x0s, dwss)
    return jnp.mean(errs) / (ts.shape[0] - 1)


def ipf_bwd_loss(f_param: JArray,
                 b_param: JArray,
                 f: Callable[[JArray, FloatScalar, JArray], JArray],
                 b: Callable[[JArray, FloatScalar, JArray], JArray],
                 xTs: JArray,
                 ts: JArray,
                 sigma: FloatScalar,
                 key: JKey) -> JFloat:
    """Mean-matching iterative proportional fitting (the backward loss).

    It's just the forward loss by swapping `f` and `b`.

    See the docstring of `ipf_fwd_loss`. Note that `ts = (t_T, t_{T-1}, ..., t_0) = T - ts = ts[::-1]`
    """
    return ipf_fwd_loss(f_param, b_param, b, f, xTs, ts, sigma, key)


def ipf(f0, f, b, f_param, b_param, x0s, xTs, ts, sigma, key):
    """

    Parameters
    ----------
    f : Callable (..., d), (), (p, ) -> (..., d)
        A parametric function for the forward process, taking three arguments: state, time, and parameters.
    b
    f_param
    b_param
    x0s
    xTs
    ts
    sigma
    key

    Returns
    -------

    """


def simulate_discrete_time(f: Callable[[JArray, FloatScalar, ...], JArray],
                           x0s: JArray,
                           ts: JArray,
                           sigma: FloatScalar,
                           key: JKey,
                           *args, **kwargs) -> JArray:
    """Simulate a discrete-time process

    .. math::

        X_k = f(X_{k-1}, t_{k-1}) + sigma \, Q_{k},  Q_{k} ~ N(0, dt).

    Parameters
    ----------
    f : Callable (d, ), (), *args, **kwargs -> (d, )
        The function `f` in the above equation. This function takes two arguments: state and time.
        The function `f` can take additional arguments `*args` and `**kwargs`.
        It models the drift of the process.
    x0s : JArray (n, d)
        The initial samples.
    ts : JArray (nsteps + 1, )
        The times `t_0, t_1, ..., t_T`.
    sigma : FloatScalar
        The dispersion term of the process.
    key : JKey
        A JAX random key.

    Returns
    -------
    JArray (n, nsteps, d)
        Trajectories at `t_1, t_2, ...t_T`.
    """
    n, d = x0s.shape
    dts = jnp.abs(jnp.diff(ts))

    def scan_body(carry, elem):
        x = carry
        t, q = elem

        x = f(x, t, *args, **kwargs) + sigma * q
        return x, x

    _, subkey = jax.random.split(key)
    dwss = jnp.sqrt(dts)[None, :, None] * jax.random.normal(subkey, (n, dts.shape[0], d))
    return jax.vmap(lambda x0, dws: jax.lax.scan(scan_body, x0, (ts[:-1], dws))[1], in_axes=[0, 0])(x0s, dwss)
