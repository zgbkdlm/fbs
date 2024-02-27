"""The Gibbs sampler.
"""
import math
import jax
import jax.numpy as jnp
from fbs.sdes.simulators import doob_bridge_simulator
from fbs.filters.csmc.csmc import csmc_kernel
from fbs.filters.csmc.resamplings import killing
from fbs.filters.smc import bootstrap_filter
from fbs.filters.resampling import stratified
from fbs.sdes.linear import StationaryLinLinearSDE
from fbs.data.base import Dataset
from fbs.typings import JKey, JArray
from typing import Callable, Tuple


def bridge_sampler(key, y0, yT, ts, sde):
    return doob_bridge_simulator(key, sde, y0, yT, ts, integration_nsteps=100, replace=True)


def gibbs_init(key, y0, x0_shape, ts,
               fwd_sampler: Callable, dataset,
               transition_sampler, likelihood_logpdf,
               nparticles, method: str = 'filter',
               x0=None):
    """Initialise the Gibbs sampler with a draw from a bootstrap filter.

    Notes
    -----
    This implementation assumes that the forward noising process is separable. If not, you can draw a bridge of y
    by yourself and change this implementation. It also assumes that the terminal x and y are independent N(0, 1).
    """
    if x0 is None:
        x0 = jnp.ones_like(x0_shape)
    key_fwd, key_u0, key_bf, key_fwd2 = jax.random.split(key, num=4)

    path_xy = fwd_sampler(key_fwd, x0, y0)
    _, path_y = dataset.unpack(path_xy)

    vs = path_y[::-1]

    def init_sampler(*_):
        """Assume uT and vT independent N(0, 1) at T."""
        return jax.random.normal(key_u0, (nparticles, *x0_shape))

    if method == 'filter':
        approx_x0 = bootstrap_filter(transition_sampler, likelihood_logpdf, vs, ts, init_sampler, key_bf, nparticles,
                                     stratified, log=True, return_last=True)
        approx_us_star = dataset.unpack(fwd_sampler(key_fwd2, approx_x0, y0))[0][::-1]
    elif method == 'smoother':
        uss = bootstrap_filter(transition_sampler, likelihood_logpdf, vs, ts, init_sampler, key_bf, nparticles,
                               stratified, log=True, return_last=False)
        approx_x0 = uss[-1]
        approx_us_star = smoother(uss, transition_logpdf)
    else:
        raise ValueError(f"Unknown method {method}")
    return approx_x0, approx_us_star


def gibbs_smoother_init():
    """Compared to `gibbs_bootstrap_init`, this should give an initialisation closer to the stationary phase.
    """
    pass


def gibbs_kernel(key: JKey, x0: JArray, y0: JArray, us_star: JArray, bs_star: JArray,
                 ts: JArray, fwd_sampler: Callable, sde: StationaryLinLinearSDE, dataset: Dataset,
                 nparticles: int,
                 transition_sampler: Callable, transition_logpdf: Callable, likelihood_logpdf: Callable,
                 doob: bool = True) -> Tuple[JArray, JArray, JArray, JArray]:
    """Gibbs kernel for our forward-backward conditional sampler.
    The carry variables are `x0`, `us_star`, and `bs_star`.

    Parameters
    ----------
    key: JKey
        The random key.
    x0: JArray (...)
        The initial state.
    y0: JArray (...)
        The observation to be conditioned on.
    us_star: JArray (nsteps + 1, ...)
        The backward filtering trajectory.
    bs_star: JArray (nsteps + 1, ...)
        The backward filtering indices.
    ts: JArray (nsteps + 1, )
        The times.
    fwd_sampler: Callable
        The forward noising sampler.
    sde: StationaryLinLinearSDE
        A linear SDE instance.
    dataset: Dataset
        The dataset.
    nparticles: int
        The number of particles.
    transition_sampler
    transition_logpdf
    likelihood_logpdf
    doob: bool, default=True
        Whether to use the Doob's diffusion bridge to generate new path of `y`.

    Returns
    -------
    JArray (...), JArray (nsteps + 1, ...), JArray (nsteps + 1, ), JArray (nsteps + 1, )
        The new x0, us_star, bs_star, and bools.
    """
    key_fwd, key_csmc, key_bridge = jax.random.split(key, num=3)
    path_xy = fwd_sampler(key_fwd, x0, y0)
    path_x, path_y = dataset.unpack(path_xy)
    us = path_x[::-1]
    vs = bridge_sampler(key_fwd, path_y[0], path_y[-1], ts, sde)[::-1] if doob else path_y[::-1]

    def init_sampler(*_):
        return us[0] * jnp.ones((nparticles, *us.shape[1:]))

    def init_likelihood_logpdf(*_):
        return -math.log(nparticles) * jnp.ones(nparticles)

    us_star_next, bs_star_next = csmc_kernel(key_csmc,
                                             us_star, bs_star,
                                             vs, ts,
                                             init_sampler, init_likelihood_logpdf,
                                             transition_sampler, transition_logpdf,
                                             likelihood_logpdf,
                                             killing, nparticles,
                                             backward=True)
    x0_next = us_star_next[-1]
    return x0_next, us_star_next, bs_star_next, bs_star_next != bs_star
