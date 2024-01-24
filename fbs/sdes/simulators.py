import jax
import jax.numpy as jnp
from fbs.typings import JKey, JArray
from typing import Callable


def reverse_simulator(key: JKey, u0: JArray, ts: JArray,
                      score: Callable, drift: Callable, dispersion: Callable,
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
        return euler_maruyama(key, u0, ts, reverse_drift, reverse_dispersion)
    else:
        raise NotImplementedError(f'Integrator {integrator} not implemented.')


def euler_maruyama(key: JKey, x0: JArray, ts: JArray,
                   drift: Callable, dispersion: Callable) -> JArray:
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

    Returns
    -------
    JArray (d, )
        The terminal value at :math:`t_n`.
    """

    def scan_body(carry, elem):
        x = carry
        rnd, t_next, t = elem

        dt = t_next - t
        x = x + drift(x, t) * dt + dispersion(t) * jnp.sqrt(dt) * rnd
        return x, None

    rnds = jax.random.normal(key, (ts.shape[0] - 1, *x0.shape))
    return jax.lax.scan(scan_body, x0, (rnds, ts[1:], ts[:-1]))[0]


def runge_kutta(key: JKey, x0: JArray, ts: JArray,
                drift: Callable, dispersion: Callable):
    pass


def multilevel_euler_maruyama(key: JArray, x0: JArray, t0: float, T: float, max_level: int,
                              drift: Callable, dispersion: Callable) -> JArray:
    pass
