import jax
import jax.numpy as jnp
from fbs.typings import JArray, JKey, JFloat, FloatScalar
from typing import Callable, Tuple, Any, Optional


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


def ipf_loss(reverse_param: JArray,
             reverse_fn: Callable[[JArray, FloatScalar, JArray], JArray],
             forward_fn: Callable[[JArray, FloatScalar, JArray], JArray],
             forward_param: JArray,
             x0s: JArray,
             ts: JArray,
             sigma: FloatScalar,
             key: JKey) -> JFloat:
    """Mean-matching iterative proportional fitting. Algorithm 1. of de Bortoli et al., 2021.

    Suppose that we have sample paths `{X_t}` from `X_0 ~ p0` and `X_T ~ pT`. We can simulate the paths by the function
    `forward_fn`. This function aims to learn the parameter `reverse_param` of the function `reverse_fn` such that this
    reverse function defines an SDE {Y_t} that `Y_0 ~ pT` and `Y_T ~ p0`.

    Parameters
    ----------
    reverse_param : JArray (p, )
        An array of parameters for the function `b`. This is the parameter to be learnt.
    forward_param : JArray (p, )
        An array of parameters for the function `f`.
    forward_fn : Callable (..., d), (), (p, ) -> (..., d)
        A parametric function for the forward process, taking three arguments: state, time, and parameters.
    reverse_fn : Callable (..., d), (), (p, ) -> (..., d)
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
    dts = jnp.abs(jnp.diff(ts))
    n, d = x0s.shape

    def init_scan(carry, elem):
        x, err = carry
        t, t_next, dw = elem

        x_next = forward_fn(x, t, forward_param) + sigma * dw
        err = err + jnp.sum((reverse_fn(x_next, t_next, reverse_param) - (
                x_next + forward_fn(x, t, forward_param) - forward_fn(x_next, t, forward_param))) ** 2)
        return (x, err), None

    key, subkey = jax.random.split(key)
    dwss = jnp.sqrt(dts)[None, :, None] * jax.random.normal(subkey, (n, dts.shape[0], d))
    errs = jax.vmap(lambda x0, dws: jax.lax.scan(init_scan, (x0, 0.), (ts[:-1], ts[1:], dws))[0][1],
                    in_axes=[0, 0])(x0s, dwss)
    return jnp.mean(errs) / (ts.shape[0] - 1)


def ipf_loss_disc(param: JArray,
                  simulator_param: JArray,
                  x0s: JArray,
                  ks: JArray,
                  gammas: FloatScalar,
                  parametric_fn: Callable[[JArray, FloatScalar, JArray], JArray],
                  simulator_fn: Callable[[JArray, FloatScalar, JArray], JArray],
                  key: JKey) -> JFloat:
    nsamples, d = x0s.shape
    nsteps = ks.shape[0] - 1

    def scan_body(carry, elem):
        x, err = carry
        k, k_next, gamma, rnd = elem

        x_next = simulator_fn(x, k, simulator_param) + jnp.sqrt(gamma) * rnd
        err = err + jnp.mean((parametric_fn(x_next, k_next, param) - (
                x_next + simulator_fn(x, k, simulator_param) - simulator_fn(x_next, k, simulator_param))) ** 2)
        return (x, err), None

    key, subkey = jax.random.split(key)
    rnds = jax.random.normal(subkey, (nsteps, nsamples, d))
    (_, err_final), _ = jax.lax.scan(scan_body, (x0s, 0.), (ks[:-1], ks[1:], gammas, rnds))
    return jnp.mean(err_final)


def ipf_loss_cont(key: JKey,
                  param: JArray,
                  simulator_param: JArray,
                  init_samples: JArray,
                  ts: JArray,
                  parametric_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                  simulator_drift: Callable[[JArray, FloatScalar, JArray], JArray]) -> JFloat:
    r"""Proposition 29, de Bortoli et al., 2021.

    Forward
    .. math::

        X_{k+1} = X_k + f(k, X_k) \delta_k / 2 + \xi_k.

    Backward
    .. math::

        X_k = X_{k+1} - b(k+1, X_{k+1}) \delta_k / 2 + \zeta_k,

    where :math:`\delta_k = \lvert t_{k+1} - t_k \rvert`.

    Notes
    -----
    Note the weird square root two in the dispersion, though in principle it can merge in the neural network.
    """
    nsteps = ts.shape[0] - 1
    fn = lambda x, t, dt: x + simulator_drift(x, t, simulator_param) * dt * 0.5

    def scan_body(carry, elem):
        x, err = carry
        t, t_next, rnd = elem

        dt = t_next - t
        x_next = x + simulator_drift(x, t, simulator_param) * dt * 0.5 + jnp.sqrt(jnp.abs(dt)) * rnd
        err = err + jnp.mean(
            (-parametric_drift(x_next, t_next, param) * dt * 0.5 - (fn(x, t, dt) - fn(x_next, t, dt))) ** 2)
        return (x, err), None

    key, subkey = jax.random.split(key)
    rnds = jax.random.normal(subkey, (nsteps, *init_samples.shape))
    (_, err_final), _ = jax.lax.scan(scan_body, (init_samples, 0.), (ts[:-1], ts[1:], rnds))
    return jnp.mean(err_final)


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


def simulate_discrete_time(f: Callable,
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
