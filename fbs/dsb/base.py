import jax
import jax.numpy as jnp
from fbs.typings import JArray, JKey, JFloat
from typing import Callable, Tuple


# def ipf_mm(x0s: JArray, xTs: JArray,
#            f0: Callable,
#            f: Callable, f_param: JArray,
#            b: Callable, b_param: JArray,
#            key: JKey,
#            ts: JArray, t0: float = 0.,
#            niters: int = 2) -> Tuple[JArray, JArray]:
#     """Mean-matching iterative proportional fitting.
#
#     Algorithm 1. of de Bortoli et al., 2021.
#
#     Parameters
#     ----------
#     x0s : JArray (n, d)
#         The samples at the initial side.
#     xTs : JArray (n, d)
#         The samples at the terminal side.
#     f0 : Callable (..., d), () -> (..., d)
#         The discrete-time drift of the reference SDE.
#         For instance, if using Euler, then f0(x, t) = x + drift(x, t) * dt.
#     f : Callable (..., d), (), (p, ) -> (..., d)
#         A parametric function for the forward process, taking three arguments: state, time, and parameters.
#     f_param : JArray (p, )
#         The parameters for the function `f`.
#     b : Callable (..., d), (), (p, ) -> (..., d)
#         A parametric function for the backward process, taking three arguments: state, time, and parameters.
#     b_param : JArray (p, )
#         The parameters for the function `f`.
#     key : JKey
#         A JAX random key
#     ts : JArray (nsteps, )
#         Times.
#     t0 : float, default=0.
#         The initial time
#     niters : int, default=2
#         How many SB iterations.
#
#     Returns
#     -------
#     JArray (p, ), JArray (p, )
#         Two arrays carrying the parameters for `f` and `b`.
#
#     Notes
#     -----
#     Double-check the annoying square two in the dispersion.
#     """
#     dts = jnp.diff(ts, prepend=t0)
#     n, d = x0s.shape
#     nsteps = ts.shape[0]
#
#     # Initial
#     def init_loss(_f_param, _b_param, _key):
#         def init_scan(carry, elem):
#             x, err = carry
#             t, dw = elem
#
#             x_next = f0(x, t) + dw
#             err = err + jnp.sum((b(x_next, t, _b_param) - (x_next + f(x, t, _f_param) - f(x_next, t, _f_param))) ** 2)
#             return (x, err), None
#
#         _key, _subkey = jax.random.split(_key)
#         dws = jnp.sqrt(dts) * jax.random.normal(_subkey, (nsteps, d))
#         errs = jax.vmap(lambda x0: jax.lax.scan(init_scan, (x0, 0.), (ts, dws))[0][1], in_axes=[0])(x0s)
#         return jnp.mean(errs)


def make_ipf_loss(x0s: JArray,
                  f: Callable,
                  b: Callable,
                  ts: JArray,
                  t0: float = 0.) -> Callable[[JArray, JArray, JKey], JFloat]:
    """Mean-matching iterative proportional fitting.

    Algorithm 1. of de Bortoli et al., 2021.

    Parameters
    ----------
    x0s : JArray (n, d)
        The samples at the initial side.
    f : Callable (..., d), (), (p, ) -> (..., d)
        A parametric function for the forward process, taking three arguments: state, time, and parameters.
    b : Callable (..., d), (), (p, ) -> (..., d)
        A parametric function for the backward process, taking three arguments: state, time, and parameters.
    ts : JArray (nsteps, )
        Times.
    t0 : float, default=0.
        The initial time

    Returns
    -------
    JArray (p, ), JArray (p, )
        Two arrays carrying the parameters for `f` and `b`.

    Notes
    -----
    Double-check the annoying square two in the dispersion.
    """
    dts = jnp.diff(ts, prepend=t0)
    n, d = x0s.shape
    nsteps = ts.shape[0]

    def loss_fn(f_param, b_param, key):
        def init_scan(carry, elem):
            x, err = carry
            t, dw = elem

            x_next = f(x, t, f_param) + dw
            err = err + jnp.sum((b(x_next, t, b_param) - (x_next + f(x, t, f_param) - f(x_next, t, f_param))) ** 2)
            return (x, err), None

        key, subkey = jax.random.split(key)
        dws = jnp.sqrt(dts) * jax.random.normal(subkey, (nsteps, d))
        errs = jax.vmap(lambda x0: jax.lax.scan(init_scan, (x0, 0.), (ts, dws))[0][1], in_axes=[0])(x0s)
        return jnp.mean(errs)

    return loss_fn
