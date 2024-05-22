"""The Gibbs sampler.
"""
import math
import jax
import jax.numpy as jnp
from fbs.sdes.simulators import doob_bridge_simulator
from fbs.samplers.csmc.csmc import csmc_kernel
from fbs.samplers.csmc.csmc import forward_pass as csmc_fwd
from fbs.samplers.csmc.resamplings import killing
from fbs.samplers.smc import bootstrap_filter, bootstrap_backward_smoother
from fbs.samplers.resampling import stratified
from fbs.sdes.linear import LinearSDE
from fbs.typings import JKey, JArray, FloatScalar, JInt, JFloat
from typing import Callable, Tuple, Optional


def bridge_sampler(key, y0, yT, ts, sde):
    """Sampling Doob's h-transform.
    """
    return doob_bridge_simulator(key, sde, y0, yT, ts, integration_nsteps=100, replace=True)


def gibbs_init(key, y0, x0_shape, ts,
               fwd_sampler: Callable, sde, unpack,
               transition_sampler, transition_logpdf, likelihood_logpdf,
               nparticles, method: str = 'smoother',
               marg_y: bool = True,
               x0=None,
               **kwargs):
    """Initialise the Gibbs sampler with a draw from a bootstrap filter/smoother.

    Notes
    -----
    This implementation assumes that the forward noising process is separable. If not, you can draw a bridge of y
    by yourself and change this implementation. It also assumes that the terminal x and y are independent N(0, 1).
    """
    if x0 is None:
        x0 = jnp.zeros(x0_shape)
    key_fwd, key_bridge, key_u0, key_bf, key_fwd2, key_bwd = jax.random.split(key, num=6)

    path_xy = fwd_sampler(key_fwd, x0, y0, **kwargs)
    _, path_y = unpack(path_xy, **kwargs)

    vs = bridge_sampler(key_bridge, path_y[0], path_y[-1], ts, sde)[::-1] if marg_y else path_y[::-1]

    def init_sampler(*_):
        """Assume uT and vT independent N(0, 1) at T."""
        return jax.random.normal(key_u0, (nparticles, *x0_shape))

    if method == 'filter':
        approx_x0 = bootstrap_filter(transition_sampler, likelihood_logpdf, vs, ts, init_sampler, key_bf, nparticles,
                                     stratified, log=True, return_last=True, **kwargs)[0][0]
        approx_us_star = unpack(fwd_sampler(key_fwd2, approx_x0, y0, **kwargs), **kwargs)[0][::-1]
    elif method == 'smoother':
        uss = bootstrap_filter(transition_sampler, likelihood_logpdf, vs, ts, init_sampler, key_bf, nparticles,
                               stratified, log=True, return_last=False, **kwargs)[0]
        approx_x0 = uss[-1, 0]
        approx_us_star = bootstrap_backward_smoother(key_bwd, uss, vs, ts, transition_logpdf, **kwargs)
    elif method == 'debug':
        approx_x0 = bootstrap_filter(transition_sampler, likelihood_logpdf, vs, ts, init_sampler, key_bf, nparticles,
                                     stratified, log=True, return_last=False, **kwargs)[0]
        approx_us_star = None
    else:
        raise ValueError(f"Unknown method {method}")
    return approx_x0, approx_us_star


def gibbs_kernel(key: JKey, x0: JArray, y0: JArray, us_star: JArray, bs_star: JArray,
                 ts: JArray,
                 fwd_sampler: Callable[[JKey, JArray, JArray, Optional], JArray],
                 sde: LinearSDE,
                 unpack: Callable,
                 nparticles: int,
                 transition_sampler: Callable,
                 transition_logpdf: Callable,
                 likelihood_logpdf: Callable,
                 marg_y: bool = False,
                 explicit_backward: bool = True,
                 explicit_final: bool = False,
                 **kwargs) -> Tuple[JArray, JArray, JArray, JArray]:
    """Gibbs kernel for our forward-backward conditional sampler.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    x0 : JArray (...)
        The initial state (any shape).
    y0 : JArray (...)
        The observation (any shape) to be conditioned on.
    us_star : JArray (nsteps + 1, ...)
        The backward filtering trajectory. Legacy parameter, not used.
    bs_star : JArray (nsteps + 1, ...)
        The backward filtering indices.
    ts : JArray (nsteps + 1, )
        The times `t_0, t_1, ... t_{nsteps}`.
    fwd_sampler : Callable (JKey, ..., ..., **kwargs)
        The forward noising sampler, which takes three arguments: random key, x0, and y0. The output is a trajectory of
        x and y.
    sde : StationaryLinLinearSDE
        A linear SDE instance.
    unpack : Callable
        A function that splits (X, Y). In the simplest case, where the joint is a concatenation of X and Y,
        then `unpack` is just `jnp.concatenate`. However, for image super-resolution or inpainting, this can be more
        complicated. See `fbs.data.images` for how we implemented `unpack` for the image tasks.
    nparticles : int
        The number of particles.
    transition_sampler : Callable (n, du), (dv, ), (), JKey -> (n, du)
        The transition sampler of `p(u_{k} | u_{k-1}, v_{k-1}, t_{k-1})` of the discretised backward SDE.
    transition_logpdf : Callable (du, ), (n, du), (dv, ), () -> (n, du)
        The logpdf of the transition distribution.
    likelihood_logpdf : Callable (dv, ), (n, du), () -> (n, )
        The logpdf of the likelihood model `p(v_{k} | u_{k-1}, v_{k-1}, t_{k-1})`.
    marg_y : bool, default=False
        Whether to use the Doob's diffusion bridge to marginalise out the path of `y`. Not used in our paper.
    explicit_backward : bool, default=True
        Whether do the backward sampling in CSMC explicitly.
    explicit_final : bool, default=False
        Whether to use the explicit reference distribution to initialise the backward CSMC.

    Returns
    -------
    JArray (...), JArray (nsteps + 1, ...), JArray (nsteps + 1, ), JArray (nsteps + 1, )
        The new x0, us_star, bs_star, and bools.
    """
    key_fwd, key_csmc, key_bridge = jax.random.split(key, num=3)
    path_xy = fwd_sampler(key_fwd, x0, y0, **kwargs)
    path_x, path_y = unpack(path_xy, **kwargs)
    us = path_x[::-1]
    vs = bridge_sampler(key_bridge, path_y[0], path_y[-1], ts, sde)[::-1] if marg_y else path_y[::-1]

    if explicit_final:
        def init_sampler(key_, n_samples):
            return jax.random.normal(key_, (n_samples, *us.shape[1:]))

        def init_likelihood_logpdf(v0, u0s, v1, **kwargs):
            return likelihood_logpdf(v0, u0s, v1, ts[0], **kwargs)

    else:
        def init_sampler(*_):
            return us[0] * jnp.ones((nparticles, *us.shape[1:]))

        def init_likelihood_logpdf(*_):
            return -math.log(nparticles) * jnp.ones(nparticles)

    if explicit_backward:
        key_csmc_fwd, key_csmc_x0, key_csmc_bwd_us, key_csmc_bwd_bs = jax.random.split(key_csmc, num=4)
        _, log_ws, uss = csmc_fwd(key_csmc_fwd, us, bs_star, vs, ts, init_sampler, init_likelihood_logpdf,
                                  transition_sampler, likelihood_logpdf, killing, nparticles,
                                  **kwargs)

        idx, _ = force_move(key_csmc_x0, jnp.exp(log_ws[-1]), bs_star[-1])
        # idx = jax.random.choice(key_csmc_x0, jnp.arange(nparticles), p=jnp.exp(log_ws[-1]), axis=0)
        x0 = uss[-1, idx]
        us_star_next = unpack(fwd_sampler(key_csmc_bwd_us, x0, y0, **kwargs), **kwargs)[0][::-1]
        bs_star_next = jax.random.randint(key_csmc_bwd_bs, (us.shape[0],), minval=0, maxval=nparticles)
    else:
        us_star_next, bs_star_next = csmc_kernel(key_csmc,
                                                 us, bs_star,
                                                 vs, ts,
                                                 init_sampler, init_likelihood_logpdf,
                                                 transition_sampler, transition_logpdf,
                                                 likelihood_logpdf,
                                                 killing, nparticles,
                                                 backward=False,
                                                 **kwargs)
    x0_next = us_star_next[-1]
    return x0_next, us_star_next, bs_star_next, bs_star_next != bs_star


def force_move(key: JKey, weights: JArray, k: FloatScalar) -> Tuple[JInt, JFloat]:
    """
    Forced-move trajectory selection. The weights are assumed to be normalised already.

    Parameters
    ----------
    key:
        Random number generator key.
    weights:
        Log-weights of the particles.
    k:
        Index of the reference particle.

    Returns
    -------
    l_T:
        New index of the ancestor of the reference particle.
    alpha:
        Probability of accepting new sample.

    Notes
    -----
    Taken from https://github.com/AdrienCorenflos/particle_mala/blob/83f62f6ede504b36cc0d76932dcd670d9e16a5aa/gradient_csmc/utils/common.py#L8
    under the Apache 2.0 License. No modifications.
    """
    M = weights.shape[0]
    key_1, key_2 = jax.random.split(key, 2)

    w_k = weights[k]
    temp = 1 - w_k

    rest_weights = weights.at[k].set(0)  # w_{-k}
    threshold = jnp.maximum(1 - jnp.exp(-M), 1 - 1e-12)  # This might be problematic
    rest_weights = jax.lax.cond(w_k < threshold, lambda: rest_weights / temp,
                                lambda: jnp.full((M,), 1 / M))  # w_{-k} / (1 - w_k)

    i = jax.random.choice(key_1, M, p=rest_weights, shape=())  # i ~ Cat(w_{-k} / (1 - w_k))
    u = jax.random.uniform(key_2, shape=())
    accept = u * (1 - weights[i]) < temp  # u < (1 - w_k) / (1 - w_i)

    alpha = jnp.nansum(temp * rest_weights / (1 - weights))
    i = jax.lax.select(accept, i, k)

    return i, jnp.clip(alpha, 0, 1.)
