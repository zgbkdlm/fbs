import jax
import jax.numpy as jnp
from fbs.typings import JKey, JArray
from typing import Callable


def reverse_simulator(key: JKey, u0, ts, score, drift, dispersion, method: str):
    T = ts[-1]

    def reverse_drift(u, t):
        return -drift(u, T - t) + dispersion(T - t) ** 2 * score(u, T - t)

    def reverse_dispersion(t):
        return dispersion(T - t)

    return euler_maruyama(key, u0, ts, reverse_drift, reverse_dispersion)


def euler_maruyama(key: JKey, u0: JArray, ts: JArray, drift: Callable, dispersion: Callable) -> JArray:
    r"""Simulate a SDE using the Euler-Maruyama method.

    Parameters
    ----------
    key: JKey
        JAX random key.
    u0: JArray (d, )
        Initial value.
    ts: JArray (n + 1, )
        Times :math:`t_0, t_1, \ldots, t_n`.
    drift: Callable (d, ), float -> (d, )
        The drift function.
    dispersion: Callable float -> float
        The dispersion function.

    Returns
    -------
    JArray (d, )
        The terminal value at :math:`t_n`.
    """

    def scan_body(carry, elem):
        u = carry
        rnd, t_next, t = elem

        dt = t_next - t
        u = u + drift(u, t) * dt + dispersion(t) * jnp.sqrt(dt) * rnd
        return u, None

    rnds = jax.random.normal(key, (ts.shape[0] - 1, u0.shape[-1]))
    return jax.lax.scan(scan_body, u0, (rnds, ts[1:], ts[:-1]))[0]


def runge_kutta():
    pass


def multilevel_euler():
    pass

