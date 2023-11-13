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

        x_next = f(x, t, f_param) + dw
        err = err + jnp.sum((b(x_next, t_next, b_param) - (x_next + f(x, t, f_param) - f(x_next, t, f_param))) ** 2)
        return (x, err), None

    key, subkey = jax.random.split(key)
    dws = jnp.sqrt(dts)[:, None] * jax.random.normal(subkey, (dts.shape[0], d))
    errs = jax.vmap(lambda x0: jax.lax.scan(init_scan, (x0, 0.), (ts[:-1], ts[1:], dws))[0][1], in_axes=[0])(x0s)
    return jnp.mean(errs)


def ipf_bwd_loss(f_param: JArray,
                 b_param: JArray,
                 f: Callable[[JArray, FloatScalar, JArray], JArray],
                 b: Callable[[JArray, FloatScalar, JArray], JArray],
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


def simulate_discrete_time(f: Callable[[JArray, FloatScalar, ...], JArray],
                           x0s: JArray,
                           ts: JArray,
                           key: JKey,
                           *args, **kwargs) -> JArray:
    """Simulate a discrete-time process

    .. math::

        X_k = f(X_{k-1}, t_{k-1}) + Q_{k}

    Parameters
    ----------
    f : Callable (d, ), (), *args, **kwargs -> (d, )
    x0s : JArray (n, d)
        The initial samples.
    ts : JArray (nsteps + 1, )
        The times `t_0, t_1, ..., t_T`.
    key : JKey
        A JAX random key.

    Returns
    -------
    JArray (n, nsteps, d)
        Trajectories at `t_1, t_2, ...t_T`.
    """
    d = x0s.shape[0]
    dts = jnp.diff(ts)

    def scan_body(carry, elem):
        x = carry
        t, dw = elem

        x = f(x, t, *args, **kwargs) + dw
        return x, x

    _, subkey = jax.random.split(key)
    dws = jnp.sqrt(dts)[:, None] * jax.random.normal(subkey, (dts.shape[0], d))
    return jax.vmap(lambda x0: jax.lax.scan(scan_body, x0, (ts[:-1], dws))[1], in_axes=[0])(x0s)
