import jax
import jax.numpy as jnp
from fbs.typings import JKey, JArray
from typing import Callable


def reverse_simulator(key: JKey, u0: JArray, ts: JArray,
                      score: Callable, drift: Callable, dispersion: Callable,
                      integration_nsteps: int = 1,
                      integrator: str = 'euler-maruyama') -> JArray:
    r"""Simulate the time-reversal of an SDE.

    Parameters
    ----------
    key : JKey
        JAX random key.
    u0 : JArray (d, )
        Initial value.
    ts : JArray (n + 1, )
        Times :math:`t_0, t_1, \ldots, t_n`.
    score : Callable (..., d), float -> (..., d)
        The score function
    drift : Callable (d, ), float -> (d, )
        The drift function.
    dispersion : Callable float -> float
        The dispersion function.
    integration_nsteps : int, default=1
        The number of integration steps between each step.
    integrator : str, default='euler-maruyama'
        The integrator for solving the reverse SDE.

    Returns
    -------
    JArray (d, )
        The terminal value of the reverse process at :math:`t_n`.
    """
    T = ts[-1]

    def reverse_drift(u, t):
        return -drift(u, T - t) + dispersion(T - t) ** 2 * score(u, T - t)

    def reverse_dispersion(t):
        return dispersion(T - t)

    if integrator == 'euler-maruyama':
        return euler_maruyama(key, u0, ts, reverse_drift, reverse_dispersion,
                              integration_nsteps=integration_nsteps)
    else:
        raise NotImplementedError(f'Integrator {integrator} not implemented.')


def euler_maruyama(key: JKey, x0: JArray, ts: JArray,
                   drift: Callable, dispersion: Callable,
                   integration_nsteps: int = 1,
                   return_path: bool = False) -> JArray:
    r"""Simulate an SDE using the Euler-Maruyama method.

    Parameters
    ----------
    key : JKey
        JAX random key.
    x0 : JArray (d, )
        Initial value.
    ts : JArray (n + 1, )
        Times :math:`t_0, t_1, \ldots, t_n`.
    drift : Callable (d, ), float -> (d, )
        The drift function.
    dispersion : Callable float -> float
        The dispersion function.
    integration_nsteps : int, default=1
        The number of integration steps between each step.
    return_path : bool, default=False
        Whether return the path or just the terminal value.

    Returns
    -------
    JArray (d, ) or JArray (n + 1, d)
        The terminal value at :math:`t_n`. or the path at :math:`t_0, \ldots, t_n`.
    """
    keys = jax.random.split(key, num=ts.shape[0] - 1)

    def step(xt, t, t_next, key_):
        def scan_body_(carry, elem):
            x = carry
            rnd, t_ = elem
            x = x + drift(x, t_) * ddt + dispersion(t_) * jnp.sqrt(ddt) * rnd
            return x, None

        ddt = (t_next - t) / integration_nsteps
        rnds = jax.random.normal(key_, (integration_nsteps, *x0.shape))
        return jax.lax.scan(scan_body_, xt, (rnds, jnp.linspace(t, t_next - ddt, integration_nsteps)))[0]

    if return_path:
        def scan_body(carry, elem):
            x = carry
            key_, t_next, t = elem

            x = step(x, t, t_next, key_)
            return x, x

        path = jax.lax.scan(scan_body, x0, (keys, ts[1:], ts[:-1]))[1]
        return jnp.concatenate([x0[None, :], path], axis=0)
    else:
        def scan_body(carry, elem):
            x = carry
            key_, t_next, t = elem

            x = step(x, t, t_next, key_)
            return x, None

        return jax.lax.scan(scan_body, x0, (keys, ts[1:], ts[:-1]))[0]


def runge_kutta(key: JKey, x0: JArray, ts: JArray,
                drift: Callable, dispersion: Callable):
    pass


def multilevel_euler_maruyama(key: JArray, x0: JArray, t0: float, T: float, max_level: int,
                              drift: Callable, dispersion: Callable) -> JArray:
    pass


def discrete_time_simulator(key: JKey, x0: JArray, ts: JArray,
                            f: Callable, q: Callable) -> JArray:
    """Simulate a discrete-time state-space model
    X(t_{k+1}) = f(X(t_k), t_{k+1}, t_k) + q(t_{k+1}, t_k) w
    """

    def scan_body(carry, elem):
        x = carry
        rnd, t_next, t = elem

        x = f(x, t_next, t) + q(t_next, t) * rnd
        return x, None

    rnds = jax.random.normal(key, (ts.shape[0] - 1, *x0.shape))
    return jax.lax.scan(scan_body, x0, (rnds, ts[1:], ts[:-1]))[0]
