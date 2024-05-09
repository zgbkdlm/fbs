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
from fbs.sdes.linear import StationaryLinLinearSDE
from fbs.data.base import Dataset
from fbs.typings import JKey, JArray, FloatScalar, JInt, JFloat
from typing import Callable, Tuple


def bridge_sampler(key, y0, yT, ts, sde):
    return doob_bridge_simulator(key, sde, y0, yT, ts, integration_nsteps=100, replace=True)


def gibbs_init(key, y0, x0_shape, ts,
               fwd_sampler: Callable, sde, unpack,
               transition_sampler, transition_logpdf, likelihood_logpdf,
               nparticles, method: str = 'smoother',
               marg_y: bool = True,
               x0=None,
               **kwargs):
    """Initialise the Gibbs sampler with a draw from a bootstrap filter.

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
                 ts: JArray, fwd_sampler: Callable, sde: StationaryLinLinearSDE, unpack: Callable,
                 nparticles: int,
                 transition_sampler: Callable, transition_logpdf: Callable, likelihood_logpdf: Callable,
                 marg_y: bool = False,
                 explicit_backward: bool = True,
                 explicit_final: bool = False,
                 **kwargs) -> Tuple[JArray, JArray, JArray, JArray]:
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
    unpack
    dataset_param: Any
    nparticles: int
        The number of particles.
    transition_sampler
    transition_logpdf
    likelihood_logpdf
    marg_y: bool, default=True
        Whether to use the Doob's diffusion bridge to marginalise out the path of `y`.
    explicit_backward
    explicit_final
        Whether to use the explicit reference distribution to initialise the backward filter.

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

    # p(x_{[0, T]}, y_{(0, T]} \mid Y_0 = y)
    # Given X^j_0 \sim pi(x \mid y), do
    # 1. forward noise:
    #       X_{(0, T]}^{j}, Y_{(0, T]}^j ~ F( \cdot \mid X_0^j, Y_0=y)
    #       Clearly, then X_{[0, T]}^{j}, Y_{(0, T]}^j are distributed according to p(x_{[0, T]}, y_{(0, T]} \mid Y_0 = y)
    # 2. preliminary:
    #   We have a kernel z_{[0, T]} ~ K( cdot \mid x_{[0, T]}, Y_{(0, T]}^j) such that, if x_{[0, T]} ~ p(x_{[0, T]} \mid y_{[0, T]}), then z_{[0, T]} ~ p(x_{[0, T]} \mid y_{[0, T]})
    #   sample:
    #       We sample X^{j+1}_{[0, T]} ~ K( cdot \mid X^j_{[0, T]}, Y_{(0, T]}^j)
    #       Clearly X^{j+1}_{[0, T]}, Y_{(0, T]}^j is then distributed according to p(x_{[0, T]}, y_{[0, T]})
    # 3. discard:
    #       Keep only X^{j+1}_{0}
    # GOTO 1.

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
